# PyAudio 多麦克风音频接口驱动

一个基于PyAudio的高性能多麦克风音频采集和处理库，支持双缓冲区实现、实时音频流处理和多设备管理。

## 🎯 项目特性

- **双缓冲区实现**: 无阻塞音频数据读取，确保数据连续性
- **音频队列缓存**: 5秒滑动窗口队列，栈式保存最新音频数据并自动溢出管理
- **时间戳支持**: 每帧音频数据都带有精确的时间戳信息
- **多设备支持**: 自动检测和管理系统音频输入设备
- **多通道处理**: 支持单声道、立体声及多声道音频采集
- **实时处理**: 低延迟音频流处理和分析
- **易于集成**: 简洁的API设计，支持上下文管理器
- **格式转换**: 支持WAV格式音频文件保存

## 📦 安装依赖

```bash
pip install pyaudio numpy
```

**注意**: 在某些系统上安装PyAudio可能需要额外步骤：

- **Windows**: 可能需要安装Visual C++构建工具
- **macOS**: `brew install portaudio`
- **Linux**: `sudo apt-get install portaudio19-dev`

## 🚀 快速开始

### 1. 检测可用设备

```python
from device_detector import list_devices_simple

# 列出所有可用的音频输入设备
devices = list_devices_simple()
for i, (device_idx, name, channels) in enumerate(devices):
    print(f"{i+1}. 设备{device_idx}: {name} (最大{channels}通道)")
```

### 2. 基础音频录制

```python
from audio_interface import MultiMicAudioInterface
import time

# 创建音频接口
with MultiMicAudioInterface(
    device_index=0,      # 设备索引
    sample_rate=44100,   # 采样率
    channels=2,          # 通道数
    chunk_size=1024      # 缓冲区大小
) as audio:
    
    # 开始录音
    audio.start_recording()
    
    # 读取10秒音频数据
    for _ in range(100):  # 44100/1024 ≈ 43次/秒
        data = audio.read_audio_double_buffer(timeout=1.0)
        if data is not None:
            print(f"获得音频数据: {data.shape}")
        time.sleep(0.02)
    
    # 停止录音
    audio.stop_recording()
```

### 3. 分通道处理

```python
# 获取所有通道数据
all_channels = audio.get_all_channels_separated(timeout=1.0)
if all_channels:
    left_channel = all_channels[0]   # 左声道
    right_channel = all_channels[1]  # 右声道

# 或获取特定通道
left_data = audio.get_channel_data(0, timeout=1.0)  # 获取通道0
```

### 4. 音频队列功能

```python
# 获取最新一帧音频及其时间戳
frame, timestamp = audio.read_queue_latest_frame()
if frame is not None:
    print(f"最新音频帧时间: {timestamp}")
    print(f"音频数据: {frame.shape}")

# 获取指定时长的音频数据（最新的N秒）
audio_data, start_ts, end_ts = audio.read_queue_duration(2.0)  # 获取最新2秒
if audio_data is not None:
    print(f"时间范围: {start_ts} ~ {end_ts}")
    print(f"音频时长: {audio_data.shape[1] / audio.sample_rate:.2f}秒")

# 获取所有队列中的音频帧
frames, timestamps = audio.read_queue_all_frames()
print(f"队列中共有 {len(frames)} 帧数据")

# 查看队列状态
status = audio.get_queue_status()
print(f"队列使用率: {status['frame_count']}/{status['max_frames']}")
print(f"覆盖时长: {status['duration']:.2f}s")
```

## 📚 API 文档

### MultiMicAudioInterface 类

音频接口主类，提供完整的音频采集和处理功能。

#### 构造函数

```python
MultiMicAudioInterface(device_index=None, sample_rate=44100, channels=2, chunk_size=1024)
```

**参数**:
- `device_index` (int, 可选): 音频设备索引，None为系统默认设备
- `sample_rate` (int): 采样率，默认44100Hz
- `channels` (int): 通道数，默认2（立体声）
- `chunk_size` (int): 音频块大小，默认1024帧

#### 核心方法

##### `start_recording()`
开始音频录制。

```python
audio.start_recording()
```

##### `stop_recording()`
停止音频录制。

```python
audio.stop_recording()
```

##### `read_audio_double_buffer(timeout=1.0)`
双缓冲区读取音频数据，实现无阻塞数据获取。

**参数**:
- `timeout` (float): 等待新数据的超时时间（秒）

**返回值**:
- `numpy.ndarray`: 音频数据，形状为`(channels, samples)`，数据类型为`float32`，取值范围[-1.0, 1.0]
- `None`: 超时或无数据

**示例**:
```python
data = audio.read_audio_double_buffer(timeout=0.5)
if data is not None:
    print(f"音频数据形状: {data.shape}")
    print(f"通道数: {data.shape[0]}")
    print(f"样本数: {data.shape[1]}")
```

##### `get_channel_data(channel_index, timeout=1.0)`
获取指定通道的音频数据。

**参数**:
- `channel_index` (int): 通道索引（0开始）
- `timeout` (float): 超时时间（秒）

**返回值**:
- `numpy.ndarray`: 单通道音频数据，形状为`(samples,)`
- `None`: 超时或通道不存在

**示例**:
```python
# 获取左声道数据
left_channel = audio.get_channel_data(0)
# 获取右声道数据  
right_channel = audio.get_channel_data(1)
```

##### `get_all_channels_separated(timeout=1.0)`
获取所有通道的分离数据，避免重复调用底层读取函数。

**参数**:
- `timeout` (float): 超时时间（秒）

**返回值**:
- `list`: 包含各通道数据的列表`[ch0_data, ch1_data, ...]`
- `None`: 超时或无数据

**示例**:
```python
channels = audio.get_all_channels_separated()
if channels:
    for i, channel_data in enumerate(channels):
        print(f"通道{i}数据: {len(channel_data)}个样本")
```

##### `is_buffer_updated()`
检查缓冲区是否有新数据更新。

**返回值**:
- `bool`: True表示有新数据，False表示无更新

##### `close()`
关闭音频接口，释放资源。

```python
audio.close()
```

#### 音频队列方法

##### `read_queue_latest_frame()`
从队列中读取最新的一帧音频数据及其时间戳。

**返回值**:
- `tuple`: `(audio_data, timestamp)` 或 `(None, None)`
  - `audio_data`: 音频数据，形状为`(channels, samples)`
  - `timestamp`: 该帧的高精度时间戳（float，Unix时间）

**示例**:
```python
frame, timestamp = audio.read_queue_latest_frame()
if frame is not None:
    import datetime
    dt = datetime.datetime.fromtimestamp(timestamp)
    print(f"最新帧时间: {dt.strftime('%H:%M:%S.%f')}")
```

##### `read_queue_duration(duration)`
从队列中读取指定时长的最新音频数据。

**参数**:
- `duration` (float): 需要读取的时长（秒），最大5秒

**返回值**:
- `tuple`: `(audio_data, start_timestamp, end_timestamp)` 或 `(None, None, None)`
  - `audio_data`: 拼接后的音频数据，形状为`(channels, total_samples)`
  - `start_timestamp`: 第一帧时间戳
  - `end_timestamp`: 最后一帧时间戳

**示例**:
```python
# 获取最新3秒的音频数据
data, start_ts, end_ts = audio.read_queue_duration(3.0)
if data is not None:
    actual_duration = data.shape[1] / audio.sample_rate
    print(f"实际获取时长: {actual_duration:.2f}秒")
```

##### `read_queue_all_frames()`
从队列中读取所有音频帧（按时间顺序排列）。

**返回值**:
- `tuple`: `(frames_list, timestamps_list)` 或 `([], [])`
  - `frames_list`: 音频帧列表，每个元素形状为`(channels, samples)`
  - `timestamps_list`: 对应的时间戳列表

**示例**:
```python
frames, timestamps = audio.read_queue_all_frames()
for i, (frame, ts) in enumerate(zip(frames, timestamps)):
    print(f"帧{i}: 时间戳={ts}, 形状={frame.shape}")
```

##### `get_queue_status()`
获取队列状态信息。

**返回值**:
- `dict`: 包含以下键值的字典
  - `frame_count`: 当前队列中的帧数
  - `max_frames`: 最大帧数
  - `duration`: 当前队列覆盖的时长（秒）
  - `max_duration`: 最大时长（秒）
  - `is_full`: 队列是否已满

**示例**:
```python
status = audio.get_queue_status()
print(f"队列使用率: {status['frame_count']}/{status['max_frames']}")
print(f"时长: {status['duration']:.2f}s/{status['max_duration']}s")
print(f"状态: {'满' if status['is_full'] else '填充中'}")
```

##### `clear_queue()`
清空音频队列。

```python
audio.clear_queue()
```

#### 上下文管理器支持

```python
# 推荐使用方式，自动管理资源
with MultiMicAudioInterface() as audio:
    audio.start_recording()
    # 处理音频数据...
    # 自动调用close()
```

### 设备检测模块

#### `detect_audio_devices()`
检测系统中所有可用的音频输入设备。

**返回值**:
- `list`: 设备信息列表，每个元素为`(device_index, device_info, supported_channels)`

#### `list_devices_simple()`  
简化的设备列表，适用于用户选择。

**返回值**:
- `list`: 简化设备信息`[(device_index, device_name, max_channels)]`

#### `get_device_by_index(device_index)`
根据设备索引获取详细设备信息。

**参数**:
- `device_index` (int): 设备索引

**返回值**:
- `tuple`: `(device_info, supported_channels)` 或 `None`

## 🎵 完整使用示例

### 录制并保存WAV文件

```python
import time
import wave
import numpy as np
from audio_interface import MultiMicAudioInterface

def record_wav(filename, duration=10, device_index=None):
    """录制音频并保存为WAV文件"""
    
    with MultiMicAudioInterface(device_index=device_index) as audio:
        with wave.open(filename, 'wb') as wav_file:
            # 设置WAV文件参数
            wav_file.setnchannels(audio.channels)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(audio.sample_rate)
            
            # 开始录音
            audio.start_recording()
            
            start_time = time.time()
            while time.time() - start_time < duration:
                # 读取音频数据
                data = audio.read_audio_double_buffer(timeout=0.1)
                if data is not None:
                    # 转换格式: (channels, samples) -> (samples, channels)
                    data = data.T
                    # 转换为16-bit整数
                    data_int16 = np.clip(data * 32767, -32768, 32767).astype(np.int16)
                    # 写入文件
                    wav_file.writeframes(data_int16.tobytes())
            
            print(f"录制完成: {filename}")

# 使用示例
record_wav("test_recording.wav", duration=5)
```

### 实时音频分析

```python
import numpy as np
from audio_interface import MultiMicAudioInterface

def real_time_analysis():
    """实时音频电平监测"""
    
    with MultiMicAudioInterface(channels=2) as audio:
        audio.start_recording()
        
        try:
            while True:
                # 获取音频数据
                data = audio.read_audio_double_buffer(timeout=0.1)
                if data is not None:
                    # 计算RMS电平
                    rms_levels = np.sqrt(np.mean(data**2, axis=1))
                    
                    # 显示电平表
                    for ch, level in enumerate(rms_levels):
                        db = 20 * np.log10(level + 1e-10)  # 避免log(0)
                        bar = "█" * int(max(0, min(50, (db + 60) / 60 * 50)))
                        print(f"CH{ch}: {bar:<50} {db:6.1f}dB")
                    
                    print("\033[2A", end="")  # 光标上移2行，实现刷新效果
                        
        except KeyboardInterrupt:
            print("\n停止监测")

# 运行实时分析
real_time_analysis()
```

### 音频队列功能演示

```python
import time
import numpy as np
from datetime import datetime
from audio_interface import MultiMicAudioInterface

def demo_audio_queue():
    """演示音频队列功能"""
    
    with MultiMicAudioInterface(channels=2) as audio:
        audio.start_recording()
        
        # 等待队列填充
        print("等待队列填充...")
        time.sleep(3)
        
        # 1. 获取最新帧及时间戳
        frame, timestamp = audio.read_queue_latest_frame()
        if frame is not None:
            dt = datetime.fromtimestamp(timestamp)
            print(f"最新帧时间: {dt.strftime('%H:%M:%S.%f')[:-3]}")
            
            # 分析音频电平
            for ch in range(frame.shape[0]):
                rms = np.sqrt(np.mean(frame[ch]**2))
                db = 20 * np.log10(rms + 1e-10)
                print(f"通道{ch}电平: {db:.1f}dB")
        
        # 2. 获取最近2秒的音频数据
        data, start_ts, end_ts = audio.read_queue_duration(2.0)
        if data is not None:
            start_time = datetime.fromtimestamp(start_ts)
            end_time = datetime.fromtimestamp(end_ts)
            actual_duration = data.shape[1] / audio.sample_rate
            
            print(f"\n获取音频时长: {actual_duration:.2f}秒")
            print(f"时间范围: {start_time.strftime('%H:%M:%S.%f')[:-3]} ~ "
                  f"{end_time.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"数据大小: {data.shape}")
        
        # 3. 监控队列状态
        for i in range(5):
            status = audio.get_queue_status()
            print(f"\n队列状态 #{i+1}:")
            print(f"  帧数: {status['frame_count']}/{status['max_frames']}")
            print(f"  时长: {status['duration']:.2f}s")
            print(f"  状态: {'已满' if status['is_full'] else '填充中'}")
            
            time.sleep(1)

# 运行队列演示
demo_audio_queue()
```

### 实时队列监控

```python
def real_time_queue_monitor():
    """实时显示队列中最新音频帧的信息"""
    
    with MultiMicAudioInterface(channels=1) as audio:
        audio.start_recording()
        
        try:
            while True:
                frame, timestamp = audio.read_queue_latest_frame()
                if frame is not None:
                    # 计算音频电平
                    rms = np.sqrt(np.mean(frame**2))
                    db = 20 * np.log10(rms + 1e-10)
                    
                    # 创建电平条
                    bar_len = int(max(0, min(40, (db + 60) / 60 * 40)))
                    bar = "█" * bar_len + "░" * (40 - bar_len)
                    
                    # 格式化时间
                    dt = datetime.fromtimestamp(timestamp)
                    time_str = dt.strftime('%H:%M:%S.%f')[:-3]
                    
                    # 队列状态
                    status = audio.get_queue_status()
                    
                    print(f"\r{time_str} | {bar} {db:6.1f}dB | "
                          f"队列: {status['frame_count']:3d}帧 "
                          f"({status['duration']:.1f}s)", end="", flush=True)
                
                time.sleep(0.05)  # 20Hz更新频率
                
        except KeyboardInterrupt:
            print("\n\n监控已停止")

# 运行实时监控
real_time_queue_monitor()
```

## 🔧 高级配置

### 自定义音频参数

```python
# 高质量录音配置
audio = MultiMicAudioInterface(
    device_index=1,        # 指定设备
    sample_rate=48000,     # 48kHz采样率
    channels=4,            # 4通道
    chunk_size=2048        # 更大缓冲区
)

# 低延迟配置
audio = MultiMicAudioInterface(
    sample_rate=44100,
    channels=1,
    chunk_size=256         # 小缓冲区，低延迟
)
```

### 错误处理

```python
try:
    with MultiMicAudioInterface(device_index=999) as audio:  # 不存在的设备
        audio.start_recording()
except Exception as e:
    print(f"音频初始化失败: {e}")

# 检查数据可用性
data = audio.read_audio_double_buffer(timeout=0.5)
if data is None:
    print("警告: 音频数据超时")
```

## ⚠️ 注意事项

1. **设备兼容性**: 不同设备支持的采样率和通道数可能不同，使用前请检测设备能力
2. **缓冲区管理**: 双缓冲区设计确保数据连续性，但请及时读取避免数据丢失
3. **内存使用**: 音频数据以float32格式存储，注意内存占用
4. **线程安全**: 音频回调在独立线程中运行，内部已做线程同步处理
5. **资源释放**: 使用完毕后请调用`close()`方法或使用上下文管理器

## 🐛 故障排除

### 常见问题

**Q: 找不到音频设备**
```python
# 检查可用设备
from device_detector import detect_audio_devices
devices = detect_audio_devices()
```

**Q: 音频数据为空或异常**
- 检查设备是否被其他程序占用
- 确认设备支持指定的采样率和通道数
- 检查系统音频权限设置

**Q: 录音延迟或丢帧**
- 减小`chunk_size`以降低延迟
- 确保系统有足够的CPU资源
- 避免在音频回调中进行耗时操作

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进本项目！

---

**作者**: handsome-Druid  
**项目地址**: https://github.com/handsome-Druid/pyaudio