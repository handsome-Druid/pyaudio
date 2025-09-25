# -*- coding: utf-8 -*-
"""
音频设备检测器
检测系统中可用的音频输入设备及其支持的通道配置
"""

import pyaudio

def detect_audio_devices():
    """
    检测系统中所有可用的音频输入设备
    
    Returns:
        list: 包含设备信息的列表，每个元素为 (device_index, device_info, supported_channels)
    """
    p = pyaudio.PyAudio()
    
    input_devices = []
    
    print("="*80)
    print("音频输入设备检测")
    print("="*80)
    print(f"系统总设备数: {p.get_device_count()}")
    
    # 获取默认设备
    try:
        default_input = p.get_default_input_device_info()
        print(f"默认输入设备: {default_input['name']} (设备 {default_input['index']})")
    except:
        print("无默认输入设备")
    
    print("\n输入设备列表:")
    print("-" * 80)
    
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                
                print(f"\n📱 设备 {i}: {info['name']}")
                print(f"   最大输入通道: {info['maxInputChannels']}")
                print(f"   默认采样率: {int(info['defaultSampleRate'])}")
                
                # 测试支持的通道配置
                supported_channels = []
                test_channels = [1, 2, 4, 6, 8]
                
                for channels in test_channels:
                    if channels <= info['maxInputChannels']:
                        try:
                            # 尝试打开流测试
                            test_stream = p.open(
                                format=pyaudio.paInt16,
                                channels=channels,
                                rate=int(info['defaultSampleRate']),
                                input=True,
                                input_device_index=i,
                                frames_per_buffer=1024,
                                start=False
                            )
                            test_stream.close()
                            supported_channels.append(channels)
                        except:
                            pass
                
                if supported_channels:
                    channels_str = ", ".join(map(str, supported_channels))
                    print(f"   ✅ 支持通道: {channels_str}")
                    recommended = max(supported_channels)
                    print(f"   🎯 推荐通道: {recommended}")
                else:
                    print(f"   ❌ 无可用通道配置")
                    supported_channels = []
                
                input_devices.append((i, info, supported_channels))
                
        except Exception as e:
            print(f"设备 {i}: 检测失败 - {e}")
    
    p.terminate()
    
    print(f"\n找到 {len(input_devices)} 个可用输入设备")
    return input_devices

def get_device_by_index(device_index):
    """
    根据设备索引获取设备信息
    
    Args:
        device_index: 设备索引
        
    Returns:
        tuple: (device_info, supported_channels) 或 None
    """
    devices = detect_audio_devices()
    for idx, info, channels in devices:
        if idx == device_index:
            return info, channels
    return None

def list_devices_simple():
    """
    简单列出可用设备（用于用户选择）
    
    Returns:
        list: [(device_index, device_name, max_channels)]
    """
    devices = detect_audio_devices()
    simple_list = []
    
    print("\n" + "="*50)
    print("可用设备选择列表:")
    print("="*50)
    
    for i, (device_idx, device_info, supported_channels) in enumerate(devices):
        max_channels = max(supported_channels) if supported_channels else 0
        device_name = device_info['name']
        
        print(f"{i+1:2d}. 设备 {device_idx:2d}: {device_name} (最大 {max_channels} 通道)")
        simple_list.append((device_idx, device_name, max_channels))
    
    return simple_list

if __name__ == "__main__":
    # 检测所有设备
    devices = detect_audio_devices()
    
    if devices:
        print("\n" + "="*50)
        print("推荐配置:")
        print("="*50)
        
        # 推荐第一个有效设备
        device_idx, device_info, supported_channels = devices[0]
        recommended_channels = max(supported_channels) if supported_channels else 1
        
        print(f"推荐设备: {device_idx} - {device_info['name']}")
        print(f"推荐通道: {recommended_channels}")
        print(f"推荐采样率: {int(device_info['defaultSampleRate'])}")
        
        print("\n使用示例:")
        print(f"from audio_interface import MultiMicAudioInterface")
        print(f"interface = MultiMicAudioInterface(")
        print(f"    device_index={device_idx},")
        print(f"    channels={recommended_channels},")
        print(f"    sample_rate={int(device_info['defaultSampleRate'])}")
        print(f")")
    else:
        print("❌ 未找到可用的音频输入设备！")