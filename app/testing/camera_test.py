import sys
sys.path.append('/usr/lib/python3/dist-packages')

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import cv2
from deepface import DeepFace

print("GStreamer successfully imported!")

def main():
    Gst.init(None)

    pipeline_str = (
        "rtspsrc location=rtsp://admin:Orlando.1@192.168.86.30:554/Streaming/Channels/101 "
        "protocols=GST_RTSP_LOWER_TRANS_TCP latency=0 ! "
        "rtpjitterbuffer ! rtph264depay ! decodebin ! videoconvert ! appsink" # fakesink" # ! autovideosink"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    pipeline.set_state(Gst.State.PLAYING)

    print("Streaming Camera 6. Press Ctrl+C to stop.")

    try:
        bus = pipeline.get_bus()
        while True:
            msg = bus.timed_pop_filtered(1000, Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                print("Pipeline error or end of stream.")
                break
    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
