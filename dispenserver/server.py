import random
import time
import threading
import requests
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from IPython.display import display
from PIL import Image
from matplotlib import pyplot as plt
from io import StringIO
from collections import defaultdict
import zipfile
import tensorflow as tf
import tarfile
import sys
import six.moves.urllib as urllib
import numpy as np
import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()


index_map = label_map_util.create_category_index_from_labelmap(
    'out320/label_map.pbtxt', use_display_name=True)
pose_model = tf.saved_model.load(
    'models/ssdresnet320/exported/saved_model')
dog_model = tf.saved_model.load(
    'models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model')


global_dc = None
global_track = None
global_dc_next = {
    'dispense': None,
    'detection': None,
    'instructions': None,
}
state = 'command'


def reset():
    global global_dc
    global global_track
    global global_dc_next
    global state

    global_dc = None
    global_track = None
    global_dc_next = {
        'dispense': None,
        'detection': None,
        'instructions': None,
    }
    state = 'command'


DEBUG = False
MONITOR = False


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)

    return output_dict


def getLabelFor(label_id):
    return index_map[label_id]['name']


def doThang(frame):
    global global_dc_next
    global state

    # hacky state machine
    if state == 'command':
        next_command = random.choice(list(index_map.values()))['name']
        global_dc_next['instructions'] = 'Play %s sound' % (next_command)

        state = next_command
        return

    if state == 'dispensing':
        time.sleep(10)

        # loop back to command
        state = 'command'
        return

    # wait 2s before starting to detect
    time.sleep(2)

    # new_frame = frame

    image_np = frame.to_ndarray(format="bgr24")

    dog_dict = run_inference_for_single_image(dog_model, image_np)
    if dog_dict['detection_classes'][0] == 18 and dog_dict['detection_scores'][0] > 0.1:
        pose_dict = run_inference_for_single_image(pose_model, image_np)
        pose = getLabelFor(pose_dict['detection_classes'][0])
        score = pose_dict['detection_scores'][0]
        if DEBUG or MONITOR:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                pose_dict['detection_boxes'],
                pose_dict['detection_classes'],
                pose_dict['detection_scores'],
                index_map,
                instance_masks=None,
                use_normalized_coordinates=True,
                line_thickness=8)

        if DEBUG:
            global_dc_next['detection'] = 'detected pose %s with score %s' % (
                pose, str(score))

        if pose == state and score > 0.5:
            global_dc_next['dispense'] = 'dispensing! detected %s with score %s' % (
                pose, str(score))
            state = 'dispensing'
        else:
            print('looking for %s but detected %s with score %s' %
                  (state, pose, str(score)))
    else:
        if DEBUG:
            print('not dog %s: %s' % (
                dog_dict['detection_classes'][0], dog_dict['detection_scores'][0]))

    if DEBUG or MONITOR:
        cv2.imshow("frame", image_np)
        cv2.waitKey(1)
    # return new_frame


async def bg():
    global global_track
    while True:
        if global_track != None and global_track.next_frame != None:
            frame = global_track.next_frame

            doThang(frame)

            # return new_frame to device?
        else:
            # wait for device to connect
            await asyncio.sleep(1)

        if DEBUG:
            print("running", global_track == None)

myloop = asyncio.new_event_loop()


def f(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


t = threading.Thread(target=f, args=(myloop,))
t.start()
myloop.create_task(bg())


class VideoTransformTrack(MediaStreamTrack):
    """
    This stream just caches the latest frame to be processed.

    We could return the image that was run through object detection, but it doesn't seem to send smoothly to the device.
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()  # don't forget this!
        self.track = track
        self.next_frame = None

    async def recv(self):
        frame = await self.track.recv()
        self.next_frame = frame
        return frame


async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            global global_dc
            global_dc = channel
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        log_info("ICE connection state is %s", pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global global_track
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            local_video = VideoTransformTrack(track)
            pc.addTrack(local_video)
            global_track = local_video

            async def go():
                global global_dc
                global global_dc_next
                while True:
                    if not global_track:
                        break
                    if global_dc and global_dc_next['detection']:
                        global_dc.send(global_dc_next['detection'])
                        global_dc_next['detection'] = None

                    if global_dc and global_dc_next['dispense']:
                        global_dc.send(global_dc_next['dispense'])
                        t = threading.Thread(target=requests.get, args=(
                            'http://192.168.1.123:8080',))
                        t.start()
                        # print('dispensing done')
                        # requests.get('http://192.168.1.123:8080')
                        global_dc_next['dispense'] = None

                    if global_dc and global_dc_next['instructions']:
                        global_dc.send(global_dc_next['instructions'])
                        global_dc_next['instructions'] = None

                    await asyncio.sleep(1)
            asyncio.create_task(go())

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            cv2.destroyAllWindows()
            reset()

    # handle offer
    await pc.setRemoteDescription(offer)

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    print("shutdown")
    myloop.stop()
    print("shutherdown")


async def down(request):
    return web.FileResponse(os.path.join(ROOT, 'Down.m4a'))


async def sit(request):
    return web.FileResponse(os.path.join(ROOT, 'Sit.m4a'))


async def paw(request):
    return web.FileResponse(os.path.join(ROOT, 'Paw.m4a'))


async def stand(request):
    return web.FileResponse(os.path.join(ROOT, 'Stand.m4a'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WebRTC audio / video / data-channels demo"
    )
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    parser.add_argument("--write-audio", help="Write received audio to a file")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_get("/client.js", javascript)
    app.router.add_get("/down", down)
    app.router.add_get("/sit", sit)
    app.router.add_get("/paw", paw)
    app.router.add_get("/stand", stand)
    app.router.add_post("/offer", offer)
    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )
