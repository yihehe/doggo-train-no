// get DOM elements
var dataChannelLog = document.getElementById('data-channel'),
    iceConnectionLog = document.getElementById('ice-connection-state'),
    iceGatheringLog = document.getElementById('ice-gathering-state'),
    signalingLog = document.getElementById('signaling-state');

// peer connection
var pc = null;

// data channel
var dc = null, dcInterval = null;

const commands = ['down', 'sit', 'paw', 'stand'];

function createPeerConnection() {
    var config = {
        sdpSemantics: 'unified-plan'
    };

    pc = new RTCPeerConnection(config);

    // register some listeners to help debugging
    pc.addEventListener('icegatheringstatechange', function () {
        iceGatheringLog.textContent += ' -> ' + pc.iceGatheringState;
    }, false);
    iceGatheringLog.textContent = pc.iceGatheringState;

    pc.addEventListener('iceconnectionstatechange', function () {
        iceConnectionLog.textContent += ' -> ' + pc.iceConnectionState;
    }, false);
    iceConnectionLog.textContent = pc.iceConnectionState;

    pc.addEventListener('signalingstatechange', function () {
        signalingLog.textContent += ' -> ' + pc.signalingState;
    }, false);
    signalingLog.textContent = pc.signalingState;

    // connect audio / video
    pc.addEventListener('track', function (evt) {
        if (evt.track.kind == 'video')
            document.getElementById('video').srcObject = evt.streams[0];
        else
            document.getElementById('audio').srcObject = evt.streams[0];
    });

    return pc;
}

function negotiate() {
    return pc.createOffer().then(function (offer) {
        return pc.setLocalDescription(offer);
    }).then(function () {
        // wait for ICE gathering to complete
        return new Promise(function (resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                function checkState() {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                }
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(function () {
        var offer = pc.localDescription;

        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then(function (response) {
        return response.json();
    }).then(function (answer) {
        // document.getElementById('answer-sdp').textContent = answer.sdp;
        return pc.setRemoteDescription(answer);
    }).catch(function (e) {
        alert(e);
    });
}

function setupDataChannel(pc) {
    var time_start = null;

    function current_stamp() {
        if (time_start === null) {
            time_start = new Date().getTime();
            return 0;
        } else {
            return new Date().getTime() - time_start;
        }
    }

    dc = pc.createDataChannel('data'); // ordered by default
    dc.onclose = function () {
        clearInterval(dcInterval);
        dataChannelLog.textContent = '- close\n' + dataChannelLog.textContent;
    };
    dc.onopen = function () {
        dataChannelLog.textContent = '- open\n' + dataChannelLog.textContent;

        // ping once
        var message = 'ping ' + current_stamp();
        dataChannelLog.textContent = '> ' + message + '\n' + dataChannelLog.textContent;
        dc.send(message);

        // dcInterval = setInterval(function() {
        //     var message = 'ping ' + current_stamp();
        //     dataChannelLog.textContent = '> ' + message + '\n' + dataChannelLog.textContent;
        //     dc.send(message);
        // }, 1000);
    };
    
    // preload the commands so it can played in callback
    commands.forEach(cmd => {
        document.getElementById(cmd).load();
    });

    dc.onmessage = function (evt) {
        dataChannelLog.textContent = '< ' + evt.data + '\n' + dataChannelLog.textContent;

        commands.forEach(cmd => {
            if (evt.data.substring(0, 4) === 'Play' && evt.data.toLowerCase().includes(cmd)) {
                document.getElementById(cmd).play();
            }
        });

        if (evt.data.substring(0, 4) === 'pong') {
            var elapsed_ms = current_stamp() - parseInt(evt.data.substring(5), 10);
            dataChannelLog.textContent = ' RTT ' + elapsed_ms + ' ms\n' + dataChannelLog.textContent;
        }
    };
}

function setupVideoChannel(pc) {
    var constraints = {
        video: {
            width: 320,
            height: 320,
            // facingMode: 'user', // front camera
            facingMode: 'environment', // rear camera
        }
    };

    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        // show preview
        document.getElementById('previewVideo').srcObject = stream;

        // send feed to server
        stream.getTracks().forEach(function (track) {
            pc.addTrack(track, stream);
        });

        return negotiate();
    }, function (err) {
        alert('Could not acquire media: ' + err);
    });
}


function start() {
    document.getElementById('start').style.display = 'none';
    dataChannelLog.textContent = ""

    pc = createPeerConnection();

    setupDataChannel(pc);
    setupVideoChannel(pc)

    document.getElementById('stop').style.display = 'inline-block';
    document.getElementById('media').style.display = 'block';
    document.getElementById('unmedia').style.display = 'none';
    document.getElementById('data').style.display = 'block';
    document.getElementById('undata').style.display = 'none';
}

function stop() {
    document.getElementById('stop').style.display = 'none';
    document.getElementById('media').style.display = 'none';
    document.getElementById('unmedia').style.display = 'block';
    document.getElementById('data').style.display = 'none';
    document.getElementById('undata').style.display = 'block';

    // close data channel
    if (dc) {
        dc.close();
    }

    // close transceivers
    if (pc.getTransceivers) {
        pc.getTransceivers().forEach(function (transceiver) {
            if (transceiver.stop) {
                transceiver.stop();
            }
        });
    }

    // close local audio / video
    pc.getSenders().forEach(function (sender) {
        sender.track.stop();
    });

    // close peer connection
    setTimeout(function () {
        pc.close();
        document.getElementById('start').style.display = 'inline-block';
    }, 500);
}
