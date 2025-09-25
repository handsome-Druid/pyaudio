# -*- coding: utf-8 -*-
"""
多麦克风音频接口驱动
双缓冲区实现：先交换→读取→清空
"""

import pyaudio
import numpy as np
import threading
import time

class MultiMicAudioInterface:
    def __init__(self, device_index=None, sample_rate=44100, channels=2, chunk_size=1024):
        """
        初始化音频接口
        
        Args:
            device_index: 音频设备索引，None为默认设备
            sample_rate: 采样率，默认44100Hz
            channels: 通道数，默认2（立体声）
            chunk_size: 音频块大小，默认1024
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        # 双缓冲区实现
        self.buffer_a = np.zeros((self.channels, self.chunk_size), dtype=np.float32)
        self.buffer_b = np.zeros((self.channels, self.chunk_size), dtype=np.float32)
        self.current_read_buffer = self.buffer_a
        self.current_write_buffer = self.buffer_b
        self.buffer_lock = threading.Lock()
        
        # 控制线程运行
        self.is_running = False
        
        # 新增：缓冲区更新标志
        self.buffer_updated = threading.Event()
        
        # Initialize PyAudio
        self.pyaudio = pyaudio.PyAudio()

        # Open the stream for the device
        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 回调函数，在音频线程中自动调用"""
        try:
            # 转换音频数据 - PyAudio返回的是交错格式 [L, R, L, R, ...]
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            # 重新组织为 (frame_count, channels) 然后转置为 (channels, frame_count)
            audio_data = audio_data.reshape((frame_count, self.channels)).T
            audio_data = audio_data.astype(np.float32) / 32768.0
            
            # 确保数据大小匹配缓冲区
            if audio_data.shape[1] == self.chunk_size:
                # 更新双缓冲区
                with self.buffer_lock:
                    self.current_write_buffer[:] = audio_data
                    # 设置缓冲区更新标志
                    self.buffer_updated.set()
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

    def read_audio_double_buffer(self, timeout=1.0):
        """
        双缓冲区读取：等待新数据→先交换，后读取，后清空
        返回分离的通道数据 (channels, samples)
        
        Args:
            timeout: 等待新数据的超时时间（秒）
            
        Returns:
            numpy.ndarray: 音频数据 (channels, samples) 或 None（超时）
        """
        # 等待缓冲区更新
        if not self.buffer_updated.wait(timeout):
            return None  # 超时
            
        with self.buffer_lock:
            # 清除更新标志
            self.buffer_updated.clear()
            
            # 1. 先交换缓冲区
            self.current_read_buffer, self.current_write_buffer = \
                self.current_write_buffer, self.current_read_buffer
            
            # 2. 读取交换后的读缓冲区数据（已经是分离的通道格式）
            data = self.current_read_buffer.copy()
            
            # 3. 清空读缓冲区，为下次交换做准备
            self.current_read_buffer.fill(0)
            
            return data

    def get_channel_data(self, channel_index, timeout=1.0):
        """
        获取指定通道的音频数据
        
        Args:
            channel_index: 通道索引 (0-based)
            timeout: 等待新数据的超时时间（秒）
            
        Returns:
            numpy.ndarray: 单通道音频数据 (samples,) 或 None（超时/错误）
        """
        data = self.read_audio_double_buffer(timeout)
        if data is None:
            return None
            
        if 0 <= channel_index < self.channels:
            return data[channel_index]
        else:
            raise ValueError(f"通道索引 {channel_index} 超出范围 (0-{self.channels-1})")
    
    def get_all_channels_separated(self, timeout=1.0):
        """
        获取所有通道的分离数据（避免重复调用read_audio_double_buffer）
        
        Args:
            timeout: 等待新数据的超时时间（秒）
            
        Returns:
            list: 每个通道的音频数据列表 [ch0_data, ch1_data, ...] 或 None（超时）
        """
        data = self.read_audio_double_buffer(timeout)
        if data is None:
            return None
        return [data[i] for i in range(self.channels)]
    
    def is_buffer_updated(self):
        """检查缓冲区是否有新数据"""
        with self.buffer_lock:
            # 简单检查：看当前写缓冲区是否为全零
            return not np.allclose(self.current_write_buffer, 0)

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