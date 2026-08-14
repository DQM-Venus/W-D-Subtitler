"""媒体音轨探测与统一预处理测试。"""

import tempfile
import unittest
import wave
from pathlib import Path

import av
import numpy as np

from wd_subtitler.media_service import (
    MediaProcessingError,
    MediaWorkspace,
    choose_default_track,
    create_media_task,
    prepare_working_audio,
)
from wd_subtitler.processing_models import AudioTrackInfo


def create_test_wav(path, samples=3200):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\x00\x00" * samples)


def create_multi_track_mkv(path):
    """生成带英语和日语两条 PCM 音轨的测试容器。"""
    container = av.open(str(path), "w", format="matroska")
    streams = []
    for language, value in (("eng", 1000), ("jpn", 3000)):
        stream = container.add_stream("pcm_s16le", rate=16000)
        stream.layout = "mono"
        stream.metadata["language"] = language
        streams.append((stream, value))
    for stream, value in streams:
        frame = av.AudioFrame.from_ndarray(
            np.full((1, 3200), value, dtype=np.int16),
            format="s16",
            layout="mono",
        )
        frame.sample_rate = 16000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    container.close()


def create_video_without_audio(path):
    """生成只有视频轨、没有音轨的测试容器。"""
    container = av.open(str(path), "w", format="matroska")
    stream = container.add_stream("mpeg4", rate=1)
    stream.width = 16
    stream.height = 16
    stream.pix_fmt = "yuv420p"
    frame = av.VideoFrame.from_ndarray(
        np.zeros((16, 16, 3), dtype=np.uint8),
        format="rgb24",
    )
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


class MediaServiceTests(unittest.TestCase):
    def test_双音轨媒体默认日语且允许切换后预处理(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "双音轨.mkv"
            create_multi_track_mkv(source)
            task = create_media_task(source)

            self.assertEqual(2, len(task.tracks))
            self.assertEqual("jpn", task.selected_track.language)

            task.selected_track_index = task.tracks[0].stream_index
            with MediaWorkspace() as workspace:
                prepared = prepare_working_audio(task, workspace)
                self.assertEqual("eng", task.selected_track.language)
                self.assertTrue(prepared.exists())

    def test_无音轨视频会返回明确错误(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "无音轨.mkv"
            create_video_without_audio(source)

            with self.assertRaises(MediaProcessingError) as raised:
                create_media_task(source)

            self.assertIn("没有可用音轨", str(raised.exception))

    def test_损坏媒体会返回明确错误(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "损坏视频.mkv"
            source.write_bytes("这不是媒体文件".encode("utf-8"))

            with self.assertRaises(MediaProcessingError) as raised:
                create_media_task(source)

            self.assertIn("无法读取媒体", str(raised.exception))

    def test_默认优先选择日语音轨(self):
        tracks = (
            AudioTrackInfo(0, 0, "eng", "aac", 2),
            AudioTrackInfo(2, 1, "jpn", "aac", 2),
        )

        self.assertEqual(2, choose_default_track(tracks))

    def test_没有日语标签时选择第一条音轨(self):
        tracks = (
            AudioTrackInfo(3, 0, "", "aac", 2),
            AudioTrackInfo(5, 1, "eng", "aac", 2),
        )

        self.assertEqual(3, choose_default_track(tracks))

    def test_无需系统FFmpeg即可预处理WAV并清理工作区(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "输入.wav"
            create_test_wav(source)
            task = create_media_task(source)
            workspace = MediaWorkspace()
            workspace_path = workspace.path

            prepared = prepare_working_audio(task, workspace)

            self.assertTrue(prepared.exists())
            with wave.open(str(prepared), "rb") as audio:
                self.assertEqual(16000, audio.getframerate())
                self.assertEqual(1, audio.getnchannels())
            workspace.close()
            self.assertFalse(workspace_path.exists())


if __name__ == "__main__":
    unittest.main()
