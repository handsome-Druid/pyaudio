# PyAudio 多麦克风音频接口驱动

一个基于PyAudio的高性能多麦克风音频采集和处理库，支持双缓冲区实现、实时音频流处理和多设备管理。

## 🎯 项目特性

- **双缓冲区实现**: 无阻塞音频数据读取，确保数据连续性
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