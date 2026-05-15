import argparse
import os
import threading
import time
from queue import Empty, Queue

import cv2

from picamera2 import Picamera2

_stream_url = None
_stream_camera_id = 0
_frame_queue = None
_sender_stop = None


def _encode_stream_jpeg(frame):
    """Convert Picamera RGB/RGBA frames to OpenCV's BGR order before JPEG encoding."""
    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif channels == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.imencode(".jpg", frame)


def _stream_sender():
    """Background thread: POST latest JPEG from queue to _stream_url."""
    import urllib.request
    import time as _time

    global _sender_stop
    while _sender_stop is None or not _sender_stop.is_set():
        try:
            jpeg = _frame_queue.get(timeout=0.5)
        except Empty:
            continue
        if not _stream_url or not jpeg:
            continue
        try:
            req = urllib.request.Request(
                _stream_url.rstrip("/") + "/frame",
                data=jpeg,
                method="POST",
                headers={
                    "Content-Type": "image/jpeg",
                    "X-Camera-Id": str(_stream_camera_id),
                    "X-Capture-Time": str(_time.time()),
                },
            )
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass


def _queue_latest_frame(jpeg: bytes) -> None:
    try:
        if _frame_queue.full():
            _frame_queue.get_nowait()
        _frame_queue.put_nowait(jpeg)
    except Exception:
        pass


def get_args():
    parser = argparse.ArgumentParser(
        description="Stream raw Picamera2 footage to the laptop without inference overlays."
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--stream-url", type=str, default=None,
                        help="If set, stream frames to this URL (e.g. http://laptop:9000)")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera id sent with streamed frames (header X-Camera-Id)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    _stream_url = getattr(args, "stream_url", None) or os.environ.get("STREAM_SERVER_URL")
    _stream_camera_id = int(os.environ.get("STREAM_CAMERA_ID", str(getattr(args, "camera_id", 0))))
    show_preview = not _stream_url

    if _stream_url:
        _frame_queue = Queue(maxsize=1)
        _sender_stop = threading.Event()
        threading.Thread(target=_stream_sender, daemon=True).start()

    picam2 = Picamera2()
    controls = {}
    if args.fps:
        controls["FrameRate"] = args.fps
    config = picam2.create_preview_configuration(controls=controls, buffer_count=4)
    picam2.start(config, show_preview=show_preview)

    try:
        if _stream_url:
            while True:
                ok, jpeg = _encode_stream_jpeg(picam2.capture_array("main"))
                if ok and jpeg is not None:
                    _queue_latest_frame(jpeg.tobytes())
        else:
            while True:
                time.sleep(0.5)
    finally:
        if _sender_stop is not None:
            _sender_stop.set()
        picam2.stop()
