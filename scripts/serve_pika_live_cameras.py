#!/usr/bin/env python3
"""Serve PIKA live camera frames from the G1 over ZMQ.

Publishes the camera inputs expected by the policy:
- D405 color -> observation.images.pikaDepthCamera
- D405 depth -> observation.depths.pikaDepthCamera
- DECXIN fisheye -> observation.images.pikaFisheyeCamera
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5562)
    parser.add_argument("--d405-serial", default="315122271825")
    parser.add_argument("--fisheye-device", default="/dev/video6")
    parser.add_argument("--no-fisheye", action="store_true", help="Serve only D405 RGB/depth.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    return parser


def opencv_device_candidates(value: str) -> list[int | str]:
    candidates: list[int | str] = []
    if value.isdigit():
        candidates.append(int(value))
        candidates.append(f"/dev/video{value}")
        return candidates
    path = Path(value)
    name = path.name
    if name.startswith("video") and name.removeprefix("video").isdigit():
        candidates.append(int(name.removeprefix("video")))
    candidates.append(value)
    return candidates


def open_opencv_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    errors: list[str] = []
    for candidate in opencv_device_candidates(device):
        for use_v4l2 in (False, True):
            cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2) if use_v4l2 else cv2.VideoCapture(candidate)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                cap.set(cv2.CAP_PROP_FPS, float(fps))
                return cap
            cap.release()
            backend = "CAP_V4L2" if use_v4l2 else "default"
            errors.append(f"{candidate!r} via {backend}")
    raise RuntimeError(f"failed to open fisheye camera {device!r}; tried {', '.join(errors)}")


class PikaLiveCameraSource:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: dict[str, Any] = {}
        self._thread: threading.Thread | None = None
        self._pipeline: rs.pipeline | None = None
        self._fisheye: cv2.VideoCapture | None = None

    def start(self) -> None:
        try:
            config = rs.config()
            config.enable_device(self.args.d405_serial)
            config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.bgr8, self.args.fps)
            config.enable_stream(rs.stream.depth, self.args.width, self.args.height, rs.format.z16, self.args.fps)
            self._pipeline = rs.pipeline()
            self._pipeline.start(config)

            if not self.args.no_fisheye:
                self._fisheye = open_opencv_camera(
                    self.args.fisheye_device,
                    self.args.width,
                    self.args.height,
                    self.args.fps,
                )
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._fisheye is not None:
            self._fisheye.release()
        if self._pipeline is not None:
            self._pipeline.stop()

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        # Drop auto-exposure warmup frames.
        for _ in range(10):
            if self._stop.is_set():
                return
            self._pipeline.wait_for_frames(5000)

        while not self._stop.is_set():
            frames = self._pipeline.wait_for_frames(5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            fisheye_bgr = None
            if self._fisheye is not None:
                ok, fisheye_bgr = self._fisheye.read()
                if not ok:
                    continue
            if not color_frame or not depth_frame:
                continue

            d405_color_bgr = np.asanyarray(color_frame.get_data()).copy()
            d405_depth = np.asanyarray(depth_frame.get_data()).copy()
            if fisheye_bgr is None:
                fisheye_bgr = np.zeros_like(d405_color_bgr)
            with self._lock:
                self._latest = {
                    "stamp": time.time(),
                    "d405_color_bgr": d405_color_bgr,
                    "d405_depth_u16": d405_depth,
                    "fisheye_bgr": fisheye_bgr.copy(),
                }

    def get_encoded_frames(self) -> list[bytes]:
        with self._lock:
            latest = dict(self._latest)
        if not latest:
            raise RuntimeError("no camera frames captured yet")

        jpg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)]
        ok_color, color_jpg = cv2.imencode(".jpg", latest["d405_color_bgr"], jpg_params)
        ok_depth, depth_png = cv2.imencode(".png", latest["d405_depth_u16"])
        ok_fisheye, fisheye_jpg = cv2.imencode(".jpg", latest["fisheye_bgr"], jpg_params)
        if not (ok_color and ok_depth and ok_fisheye):
            raise RuntimeError("failed to encode camera frames")

        meta = (
            "{"
            f"\"stamp\":{latest['stamp']:.6f},"
            f"\"d405_color_shape\":{list(latest['d405_color_bgr'].shape)},"
            f"\"d405_depth_shape\":{list(latest['d405_depth_u16'].shape)},"
            f"\"fisheye_shape\":{list(latest['fisheye_bgr'].shape)},"
            f"\"fisheye_enabled\":{str(not self.args.no_fisheye).lower()}"
            "}"
        ).encode("utf-8")
        return [meta, color_jpg.tobytes(), depth_png.tobytes(), fisheye_jpg.tobytes()]


def main() -> None:
    args = build_parser().parse_args()
    source = PikaLiveCameraSource(args)
    shutdown = threading.Event()

    def _handle_signal(_signum: int, _frame: Any) -> None:
        shutdown.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    source.start()
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"PIKA live camera server listening on tcp://{args.host}:{args.port}")
    print(f"D405 serial: {args.d405_serial}")
    if args.no_fisheye:
        print("DECXIN fisheye: disabled")
    else:
        print(f"DECXIN fisheye: {args.fisheye_device}")

    try:
        while not shutdown.is_set():
            try:
                request = sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.002)
                continue
            if request == "frames":
                try:
                    sock.send_multipart(source.get_encoded_frames())
                except Exception as exc:
                    sock.send_multipart([f'{{"error":"{type(exc).__name__}: {exc}"}}'.encode("utf-8")])
            elif request == "ping":
                sock.send_string("pong")
            else:
                sock.send_string(f"unknown request: {request}")
    finally:
        source.stop()
        sock.close(linger=0)
        ctx.term()


if __name__ == "__main__":
    main()
