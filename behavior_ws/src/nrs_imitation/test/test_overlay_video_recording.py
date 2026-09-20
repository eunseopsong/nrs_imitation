"""Exercise real headless encoding with fake ROS transport; never rclpy.init()."""
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import Image

from nrs_imitation import overlay_video_recorder as recording


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="requires ffmpeg/ffprobe")
def test_default_recording_is_headless_low_rate_and_finalizes(tmp_path, monkeypatch):
    subscriptions, timers, commands = [], [], []
    params = {}
    def declare(node, key, default):
        params[key] = {"output_dir": str(tmp_path), "append_removal_heatmap": False}.get(key, default)
    def timer(node, period, callback):
        timers.append(period)
        return NS(cancel=lambda: None)
    # Run the real recorder constructor/callbacks/shutdown, substituting transport only.
    monkeypatch.setattr(recording.Node, "__init__", lambda *a, **kw: None)
    monkeypatch.setattr(recording.Node, "declare_parameter", declare)
    monkeypatch.setattr(recording.Node, "get_parameter", lambda node, key: NS(value=params[key]))
    monkeypatch.setattr(recording.Node, "get_logger", lambda node: NS(info=lambda *a: None, warn=lambda *a: None))
    monkeypatch.setattr(recording.Node, "create_subscription", lambda node, *args: subscriptions.append(args))
    monkeypatch.setattr(recording.Node, "create_timer", timer)
    monkeypatch.setattr(recording.Node, "destroy_node", lambda node: None)
    def no_gui(*args, **kwargs):
        pytest.fail("recording must not open a GUI")
    monkeypatch.setattr(cv2, "imshow", no_gui)
    real_popen = subprocess.Popen
    def popen(cmd, **kw):
        commands.append(cmd)
        return real_popen(cmd, **kw)
    monkeypatch.setattr(recording.subprocess, "Popen", popen)
    previous_threads = cv2.getNumThreads()
    node = recording.OverlayVideoRecorderNode()
    try:
        assert params["enable"] is True
        assert timers == [0.1] and node.framerate == 10.0
        assert len(subscriptions) == 2 and all(s[3].depth == 1 for s in subscriptions)
        assert cv2.getNumThreads() == 1
        for i in range(12):
            # Different panel dimensions exercise composition/padding as well.
            flow = np.full((48, 96, 3), i * 15, np.uint8)
            modality = np.full((32, 64, 3), 100, np.uint8)
            node._cb_flow(Image(height=48, width=96, encoding="rgb8", step=288, data=flow.tobytes()))
            node._cb_modality(Image(height=32, width=64, encoding="rgb8", step=192, data=modality.tobytes()))
            assert node._prev_composite is None  # no full-frame float crossfades
            node._cb_timer()
    finally:
        node.destroy_node()
        cv2.setNumThreads(previous_threads)
    assert node.proc.returncode == 0 and node._frames_written == 12
    assert Path(node.out_path).is_file()
    cmd = commands[0]
    assert cmd[cmd.index("-threads") + 1] == "1"
    if shutil.which("nice"):
        assert cmd[1:3] == ["-n", "10"]
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", node.out_path
    ], text=True))
    stream = probe["streams"][0]
    assert stream["r_frame_rate"] == "10/1"
    assert stream["nb_read_frames"] == "12"
    assert (stream["width"], stream["height"]) == (192, 48)
    assert float(probe["format"]["duration"]) == pytest.approx(1.2, abs=.02)
