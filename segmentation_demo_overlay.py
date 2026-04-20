import argparse
import os
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Dict

import cv2
import numpy as np

from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

SCRIPT_DIR = Path(__file__).resolve().parent
COLOURS = np.loadtxt(str(SCRIPT_DIR / "assets" / "colours.txt"))

_stream_url = None
_stream_camera_id = 0
_frame_queue = None
_sender_stop = None


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


def create_and_draw_masks(request: CompletedRequest):
    """Create masks from the output tensor and draw them on the main output image."""
    masks = create_masks(request)
    draw_masks(masks)

    # If streaming, composite main + overlay (if any) and send (copy frame inside context)
    if _stream_url and _frame_queue is not None:
        with MappedArray(request, "main") as m:
            main = m.array.copy()
            # Main buffer may be RGB or RGBA; use only RGB for blending and output
            main_rgb = main[:, :, :3] if main.shape[-1] == 4 else main
            if masks:
                input_w, input_h = imx500.get_input_size()
                out_h, out_w = main.shape[:2]
                output_shape = [input_h, input_w, 4]
                overlay = np.zeros(output_shape, dtype=np.uint8)
                for v in masks.values():
                    overlay += v
                overlay[:, :, 3] = 255
                if (out_h, out_w) != (input_h, input_w):
                    overlay = cv2.resize(overlay, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
                rgb = overlay[:, :, :3].astype(np.float32)
                blended = (1 - alpha) * main_rgb.astype(np.float32) + alpha * rgb
                frame = np.clip(blended, 0, 255).astype(np.uint8)
            else:
                frame = main_rgb
            _, jpeg = cv2.imencode(".jpg", frame)
            if jpeg is not None:
                try:
                    _frame_queue.put_nowait(jpeg.tobytes())
                except Exception:
                    pass


def create_masks(request: CompletedRequest) -> Dict[int, np.ndarray]:
    """Create masks from the output tensor, scaled to the ISP output."""
    res = {}
    np_outputs = imx500.get_outputs(metadata=request.get_metadata())
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        return res
    mask = np_outputs[0]
    found_indices = np.unique(mask)

    for i in found_indices:
        if i == 0:
            continue
        output_shape = [input_h, input_w, 4]
        colour = [(0, 0, 0, 0), COLOURS[int(i)]]
        colour[1][3] = 150  # update the alpha value here, to save setting it later
        overlay = np.array(mask == i, dtype=np.uint8)
        overlay = np.array(colour)[overlay].reshape(output_shape).astype(np.uint8)
        # No need to resize the overlay, it will be stretched to the output window.
        res[i] = overlay
    return res


def draw_masks(masks: Dict[int, np.ndarray]):
    """Draw the masks for this request onto the ISP output."""
    if not masks:
        return
    input_w, input_h = imx500.get_input_size()
    output_shape = [input_h, input_w, 4]
    overlay = np.zeros(output_shape, dtype=np.uint8)
    if masks:
        for v in masks.values():
            overlay += v
        # Set Alphas and overlay
        overlay[:,:,3]=255
        picam2.set_overlay(overlay)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path of the model",
                        default=str(SCRIPT_DIR / "models" / "imx500_network_deeplabv3plus.rpk"))
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--print-intrinsics", action="store_true",
                        help="Print JSON network_intrinsics then exit")
    parser.add_argument("--stream-url", type=str, default=None,
                        help="If set, stream frames to this URL (e.g. http://laptop:9000) instead of local preview")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="Camera id sent with streamed frames (header X-Camera-Id)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # This must be called before instantiation of Picamera2
    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics
    if not intrinsics:
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "segmentation"
    elif intrinsics.task != "segmentation":
        print("Network is not a segmentation task", file=sys.stderr)
        exit()

    # Override intrinsics from args
    for key, value in vars(args).items():
        if hasattr(intrinsics, key) and value is not None:
            setattr(intrinsics, key, value)

    # Defaults
    intrinsics.update_with_defaults()

    if args.print_intrinsics:
        print(intrinsics)
        exit()

    _stream_url = getattr(args, "stream_url", None) or os.environ.get("STREAM_SERVER_URL")
    show_preview = not _stream_url
    if _stream_url:
        _stream_camera_id = int(os.environ.get("STREAM_CAMERA_ID", str(getattr(args, "camera_id", 0))))
        _frame_queue = Queue(maxsize=1)
        _sender_stop = threading.Event()
        threading.Thread(target=_stream_sender, daemon=True).start()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=show_preview)
    picam2.pre_callback = create_and_draw_masks

    while True:
        time.sleep(0.5)
