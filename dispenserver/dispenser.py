#!/usr/bin/env python
# remember this is running on python2.7

import nxt.locator
from nxt.motor import *
import SimpleHTTPServer
import SocketServer
import time
import sys

MODE = 'normal'
if len(sys.argv) > 1:
    if sys.argv[1] == 'calibrate':
        MODE = 'calibrate'
    if sys.argv[1] == 'test':
        MODE = 'test'

PORT = 8080
power = 25  # 15 is less aggressive
tacho = 200

b = m = None
if MODE != 'test':
    b = nxt.locator.find_one_brick()
    m = Motor(b, PORT_A)

# calibration mode
while MODE == 'calibrate':
    p = raw_input("power: ")
    t = raw_input("tacho: ")
    m.turn(int(p), int(t))


def extend():
    print("extending")
    if MODE != 'test':
        m.turn(power, tacho)


def retract():
    print("retracting")
    if MODE != 'test':
        m.turn(-power, tacho)


class NxtHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        # this will hang
        extend()
        time.sleep(5)
        retract()
        self.send_response(200)


httpd = SocketServer.TCPServer(("", PORT), NxtHandler)

print("Serving on port", PORT)
httpd.serve_forever()

"""
Use this to turn a motor. The motor will not stop until it turns the
desired distance. Accuracy is much better over a USB connection than
with bluetooth...
power is a value between -127 and 128 (an absolute value greater than
         64 is recommended)
tacho_units is the number of degrees to turn the motor. values smaller
         than 50 are not recommended and may have strange results.
brake is whether or not to hold the motor after the function exits
         (either by reaching the distance or throwing an exception).
timeout is the number of seconds after which a BlockedException is
         raised if the motor doesn't turn
emulate is a boolean value. If set to False, the motor is aware of the
         tacho limit. If True, a run() function equivalent is used.
         Warning: motors remember their positions and not using emulate
         may lead to strange behavior, especially with synced motors
"""
