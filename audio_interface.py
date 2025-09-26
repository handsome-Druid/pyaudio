# -*- coding: utf-8 -*-
"""
多麦克风音频接口驱动（基于栈式/队列缓存）
保留最近固定时长（默认5秒）的音频帧，并提供便捷的读取接口。
"""

import pyaudio
import numpy as np
import threading
import time
from collections import deque

class MultiMicAudioInterface:
    def __init__(self, device_index=None, sample_rate=44100, channels=2, chunk_size=1024, sample_format: str = "auto"):
        """
        初始化音频接口
        
        Args:
            device_index: 音频设备索引，None为默认设备
            sample_rate: 采样率，默认44100Hz
            channels: 通道数，默认2（立体声）
            chunk_size: 音频块大小，默认1024
            sample_format: 采样格式，可选 "auto" | "int24" | "int16" | "float32"，默认自动优先尝试24位
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.sample_format_name = (sample_format or "auto").lower()
        self._pa_format = None  # 实际打开成功的PortAudio格式
        self._sample_width = None  # 每样本字节数
        # 输入数据转换函数，返回float32且形状为 (channels, frames)
        self._convert_input = None
        
        # 控制线程运行
        self.is_running = False
        
        # 音频队列功能 - 保存5秒的音频数据
        self.queue_duration = 5.0  # 5秒
        self.queue_max_frames = int(self.sample_rate * self.queue_duration / self.chunk_size)
        self.audio_queue = deque(maxlen=self.queue_max_frames)  # 使用deque实现栈式存储
        self.queue_timestamps = deque(maxlen=self.queue_max_frames)  # 对应的时间戳
        self.queue_lock = threading.Lock()
        # 新数据事件：有新帧写入时触发，便于阻塞等待最新帧
        self.new_frame_event = threading.Event()
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()

        # 打开输入流（支持自动/指定采样格式）
        self.stream = self._open_stream_with_preferred_format(device_index)

    # ---------------- 内部：打开流与格式解析 ----------------
    def _open_stream_with_preferred_format(self, device_index):
        """根据期望格式尝试打开输入流，并设置对应的解析函数。"""
        # 优先级：int24 -> float32 -> int16（若为auto）
        name2fmt = {
            "int24": pyaudio.paInt24,
            "int16": pyaudio.paInt16,
            "float32": pyaudio.paFloat32,
        }

        if self.sample_format_name == "auto":
            candidates = ["int24", "float32", "int16"]
        else:
            if self.sample_format_name not in name2fmt:
                raise ValueError(f"不支持的sample_format: {self.sample_format_name}")
            candidates = [self.sample_format_name]

        last_err = None
        for name in candidates:
            pa_fmt = name2fmt[name]
            try:
                stream = self.pyaudio.open(
                    format=pa_fmt,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
                # 成功：记录格式、样宽与转换器
                self._pa_format = pa_fmt
                self.sample_format_name = name
                self._sample_width = self.pyaudio.get_sample_size(pa_fmt)
                if name == "int24":
                    self._convert_input = self._convert_int24_to_float32
                elif name == "int16":
                    self._convert_input = self._convert_int16_to_float32
                elif name == "float32":
                    self._convert_input = self._convert_float32_to_float32
                print(f"🎧 输入流已打开，采样格式: {name}（{self._sample_width*8}位）")
                return stream
            except Exception as e:
                last_err = e
                continue

        # 全部失败
        raise RuntimeError(f"无法打开输入流，尝试的格式: {candidates}，最后错误: {last_err}")

    def _convert_int16_to_float32(self, in_data, frame_count):
        audio = np.frombuffer(in_data, dtype=np.int16)
        audio = audio.reshape((frame_count, self.channels)).T
        return (audio.astype(np.float32) / 32768.0)

    def _convert_float32_to_float32(self, in_data, frame_count):
        audio = np.frombuffer(in_data, dtype=np.float32)
        audio = audio.reshape((frame_count, self.channels)).T
        return audio

    def _convert_int24_to_float32(self, in_data, frame_count):
        # 24bit PCM → int32 有符号，再归一化到[-1,1)
        b = np.frombuffer(in_data, dtype=np.uint8)
        expected = frame_count * self.channels * 3
        if b.size != expected:
            # 尺寸异常，尽量在不崩溃的情况下返回空
            print(f"警告：接收字节数与24位期望不符: got={b.size}, expected={expected}")
            return np.zeros((self.channels, frame_count), dtype=np.float32)
        b = b.reshape(-1, 3)
        # little-endian: LSB, mid, MSB
        val = (b[:, 0].astype(np.int32) |
               (b[:, 1].astype(np.int32) << 8) |
               (b[:, 2].astype(np.int32) << 16))
        # 符号扩展（24bit）
        neg = (val & 0x800000) != 0
        val[neg] -= (1 << 24)
        # 归一化
        audio = (val.astype(np.float32) / 8388608.0)  # 2^23
        audio = audio.reshape((frame_count, self.channels)).T
        return audio

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 回调函数，在音频线程中自动调用"""
        try:
            # 将回调输入转换为 float32, 形状 (channels, frame_count)
            if self._convert_input is None:
                # 理论上不应发生
                print("错误：未初始化的输入转换器")
                return (None, pyaudio.paAbort)
            audio_data = self._convert_input(in_data, frame_count)
            
            # 确保数据大小匹配缓冲区
            if audio_data.shape[1] == self.chunk_size:
                # 获取当前时间戳（使用高精度时间）
                current_timestamp = time.time()
                # 更新音频队列（栈式存储最新数据）
                with self.queue_lock:
                    self.audio_queue.append(audio_data.copy())  # 添加到队列末尾（最新数据）
                    self.queue_timestamps.append(current_timestamp)
                    # deque会自动删除超出maxlen的旧数据
                # 通知有新帧到达
                self.new_frame_event.set()
                    
            else:
                print(f"警告：音频帧大小不匹配 {audio_data.shape} vs expected ({self.channels}, {self.chunk_size})")
                
        except Exception as e:
            print(f"音频回调错误: {e}")
            
        return (None, pyaudio.paContinue)

    def start_recording(self):
        """开始录音"""
        if not self.is_running:
            self.is_running = True
            self.stream.start_stream()

    def stop_recording(self):
        """停止录音"""
        if self.is_running:
            self.is_running = False
            self.stream.stop_stream()

    def wait_for_next_frame(self, timeout: float = 1.0):
        """
        阻塞等待下一帧的到来，并返回最新帧及其时间戳。
        如果在超时时间内没有新帧，则返回 (None, None)。
        """
        # 快速清除上一次的事件（避免旧事件造成误判）
        self.new_frame_event.clear()
        if not self.new_frame_event.wait(timeout):
            return None, None
        return self.read_queue_latest_frame()
    
    def read_queue_latest_frame(self):
        """
        从队列中读取最新的一帧音频数据
        
        Returns:
            tuple: (audio_data, timestamp) 或 (None, None)
                audio_data: 音频数据 (channels, samples)
                timestamp: 该帧的时间戳
        """
        # 仅在锁内获取引用，复制在锁外完成，避免阻塞回调线程
        with self.queue_lock:
            if len(self.audio_queue) == 0:
                return None, None
            latest_ref = self.audio_queue[-1]
            latest_timestamp = self.queue_timestamps[-1]
        # 在锁外复制数据
        return latest_ref.copy(), latest_timestamp
    
    def read_queue_all_frames(self):
        """
        从队列中读取所有音频帧（按时间顺序，最旧到最新）
        
        Returns:
            tuple: (frames_list, timestamps_list) 或 ([], [])
                frames_list: 音频帧列表，每个元素为 (channels, samples)
                timestamps_list: 对应的时间戳列表
        """
        # 锁内做快照，锁外做复制，缩短加锁时间
        with self.queue_lock:
            if len(self.audio_queue) == 0:
                return [], []
            frames_snapshot = list(self.audio_queue)
            timestamps = list(self.queue_timestamps)
        frames = [frame.copy() for frame in frames_snapshot]
        return frames, timestamps
    
    def read_queue_duration(self, duration):
        """
        从队列中读取指定时长的最新音频数据
        
        Args:
            duration: 需要读取的时长（秒），最大5秒
            
        Returns:
            tuple: (audio_data, start_timestamp, end_timestamp) 或 (None, None, None)
                audio_data: 拼接后的音频数据 (channels, total_samples)
                start_timestamp: 第一帧时间戳
                end_timestamp: 最后一帧时间戳
        """
        duration = min(duration, self.queue_duration)  # 限制最大时长
        needed_frames = int(duration * self.sample_rate / self.chunk_size)
        # 锁内仅计算区间并做快照，锁外执行耗时拼接
        with self.queue_lock:
            qlen = len(self.audio_queue)
            if qlen == 0:
                return None, None, None
            actual_frames = min(needed_frames, qlen)
            start_idx = qlen - actual_frames
            frames_snapshot = list(self.audio_queue)[start_idx:]
            timestamps_snapshot = list(self.queue_timestamps)[start_idx:]
        if not frames_snapshot:
            return None, None, None
        concatenated_audio = np.concatenate(frames_snapshot, axis=1)
        start_timestamp = timestamps_snapshot[0]
        end_timestamp = timestamps_snapshot[-1]
        return concatenated_audio, start_timestamp, end_timestamp
    
    def get_queue_status(self):
        """
        获取队列状态信息
        
        Returns:
            dict: 队列状态信息
                - frame_count: 当前队列中的帧数
                - max_frames: 最大帧数
                - duration: 当前队列覆盖的时长（秒）
                - max_duration: 最大时长（秒）
                - is_full: 队列是否已满
        """
        with self.queue_lock:
            frame_count = len(self.audio_queue)
            current_duration = frame_count * self.chunk_size / self.sample_rate
            
            return {
                'frame_count': frame_count,
                'max_frames': self.queue_max_frames,
                'duration': current_duration,
                'max_duration': self.queue_duration,
                'is_full': frame_count >= self.queue_max_frames
            }

    def get_queue_time_range(self):
        """
        获取当前缓存中的时间范围（最旧时间戳, 最新时间戳）。
        若队列为空，返回 (None, None)。
        """
        with self.queue_lock:
            if not self.queue_timestamps:
                return None, None
            return self.queue_timestamps[0], self.queue_timestamps[-1]

    def read_queue_by_time(self, start_timestamp: float, end_timestamp: float):
        """
        按时间戳范围读取缓存中的音频数据。
        要求[start_timestamp, end_timestamp]完全落在5秒固定缓存范围内，
        否则直接抛出 ValueError。

        说明：
        - 每帧的时间戳表示该帧结束的时间点（近似PyAudio回调到达时间）。
        - 本方法选择满足 start < ts <= end 的所有帧并拼接，不做样本级裁剪。

        Args:
            start_timestamp: 起始时间戳（Unix秒）
            end_timestamp: 截止时间戳（Unix秒）

        Returns:
            tuple: (audio_data, actual_start_timestamp, actual_end_timestamp)
                audio_data: (channels, total_samples)
                actual_start_timestamp: 选中第一帧的时间戳
                actual_end_timestamp: 选中最后一帧的时间戳

        Raises:
            ValueError: 参数无效、队列为空、超出缓存范围或范围内无数据
        """
        if start_timestamp is None or end_timestamp is None:
            raise ValueError("start_timestamp 和 end_timestamp 不能为空")
        if end_timestamp <= start_timestamp:
            raise ValueError("end_timestamp 必须大于 start_timestamp")

        # 锁内获取时间范围与快照
        with self.queue_lock:
            qlen = len(self.queue_timestamps)
            if qlen == 0:
                raise ValueError("音频缓存为空")

            oldest = self.queue_timestamps[0]
            newest = self.queue_timestamps[-1]

            # 边界检查：完整落入当前5秒缓存窗口
            if start_timestamp < oldest or end_timestamp > newest:
                raise ValueError(
                    f"请求的时间范围超出缓存窗口: 请求[{start_timestamp:.3f}, {end_timestamp:.3f}], 缓存[{oldest:.3f}, {newest:.3f}]"
                )

            # 做快照以缩短持锁
            ts_snapshot = list(self.queue_timestamps)
            frames_snapshot = list(self.audio_queue)

        # 锁外查找满足 start < ts <= end 的帧索引区间
        sel_idx = [i for i, ts in enumerate(ts_snapshot) if (ts > start_timestamp and ts <= end_timestamp)]
        if not sel_idx:
            raise ValueError("指定时间范围内没有可用的音频帧")

        first_i, last_i = sel_idx[0], sel_idx[-1]
        selected_frames = frames_snapshot[first_i:last_i + 1]
        selected_timestamps = ts_snapshot[first_i:last_i + 1]

        # 拼接音频（锁外）
        audio = np.concatenate(selected_frames, axis=1)
        return audio, selected_timestamps[0], selected_timestamps[-1]
    
    def clear_queue(self):
        """清空音频队列"""
        with self.queue_lock:
            self.audio_queue.clear()
            self.queue_timestamps.clear()

    def close(self):
        """关闭音频接口"""
        self.stop_recording()
        if self.stream:
            self.stream.close()
        self.pyaudio.terminate()

    def __enter__(self):
        """支持 with 语句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支持 with 语句"""
        self.close()