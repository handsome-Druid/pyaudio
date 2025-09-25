# -*- coding: utf-8 -*-
"""
音频接口测试程序
指定设备和通道数，录制30秒音频并保存为MP3
"""

import sys
import time
import numpy as np
from datetime import datetime
from audio_interface import MultiMicAudioInterface
from device_detector import list_devices_simple, detect_audio_devices

def record_to_mp3(device_index, channels, duration=30, output_filename=None):
    """
    录制音频并保存为WAV格式（流式写入，内存友好）
    
    Args:
        device_index: 音频设备索引
        channels: 通道数
        duration: 录制时长（秒）
        output_filename: 输出文件名，None时自动生成
    
    Returns:
        str: 保存的文件路径
    """
    import time
    import wave
    from datetime import datetime
    
    # 生成文件名
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"audio_record_{timestamp}_dev{device_index}_ch{channels}.wav"
    
    print(f"📡 开始录制音频...")
    print(f"   设备索引: {device_index}")
    print(f"   通道数: {channels}")
    print(f"   时长: {duration} 秒")
    print(f"   输出文件: {output_filename}")
    
    # 创建音频接口和WAV文件
    with MultiMicAudioInterface(
        device_index=device_index,
        channels=channels,
        sample_rate=44100,
        chunk_size=1024
    ) as audio_interface, wave.open(output_filename, 'wb') as wav_file:
        
        # 设置WAV文件参数
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(44100)
        
        # 开始录音
        audio_interface.start_recording()
        
        # 流式录制音频数据
        start_time = time.time()
        frame_count = 0
        total_samples_written = 0
        
        print(f"🎙️  录音中... (预计 {duration} 秒)")
        
        while time.time() - start_time < duration:
            # 等待新的音频数据（同步读取）
            audio_data = audio_interface.read_audio_double_buffer(timeout=0.1)
            
            if audio_data is None:
                # 超时，继续等待
                continue
            
            # 转换格式并直接写入文件
            # 从 (channels, samples) 转换为 (samples, channels)
            audio_data = audio_data.T
            # 转换到int16并写入
            audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())
            
            frame_count += 1
            total_samples_written += audio_data.shape[0]  # 实际写入的样本数
            
            # 每10块显示一次进度
            elapsed = time.time() - start_time
            if frame_count % 10 == 0:
                remaining = duration - elapsed
                progress = (elapsed / duration) * 100
                print(f"   进度: {progress:5.1f}% | 已录制: {elapsed:5.1f}s | 剩余: {remaining:5.1f}s | 块数: {frame_count}", end="\r")
        
        print(f"\n✅ 录制完成！共录制 {frame_count} 个音频块，{total_samples_written} 个音频帧")
    
    # 显示文件信息
    try:
        import os
        file_size = os.path.getsize(output_filename)
        actual_duration = total_samples_written / 44100
        print(f"💾 音频已保存到: {output_filename}")
        print(f"   文件大小: {file_size:,} 字节 ({file_size/1024/1024:.2f} MB)")
        print(f"   实际时长: {actual_duration:.2f} 秒")
        return output_filename
    except Exception as e:
        print(f"❌ 保存文件时出错: {e}")
        return None


def main():
    """主程序"""
    print("="*60)
    print("🎵 音频录制测试程序")
    print("="*60)
    
    # 检测可用设备
    try:
        devices = list_devices_simple()
        if not devices:
            print("❌ 未找到可用的音频输入设备！")
            return
            
    except Exception as e:
        print(f"❌ 检测音频设备时出错: {e}")
        return
    
    # 用户选择设备
    try:
        print(f"\n请选择音频设备 (1-{len(devices)}):")
        choice = input("设备编号 (默认1): ").strip()
        
        if choice == "":
            device_choice = 0
        else:
            device_choice = int(choice) - 1
            
        if not (0 <= device_choice < len(devices)):
            print("❌ 无效的设备选择")
            return
            
        device_index, device_name, max_channels = devices[device_choice]
        print(f"✅ 选择设备: {device_name} (设备 {device_index})")
        
    except (ValueError, KeyboardInterrupt):
        print("\n❌ 用户取消或输入无效")
        return
    
    # 选择通道数
    try:
        print(f"\n该设备最大支持 {max_channels} 通道")
        channels_input = input(f"请输入通道数 (1-{max_channels}, 默认{min(2, max_channels)}): ").strip()
        
        if channels_input == "":
            channels = min(2, max_channels)
        else:
            channels = int(channels_input)
            
        if not (1 <= channels <= max_channels):
            print(f"❌ 通道数必须在 1-{max_channels} 之间")
            return
            
        print(f"✅ 选择通道数: {channels}")
        
    except (ValueError, KeyboardInterrupt):
        print("\n❌ 用户取消或输入无效")
        return
    
    # 选择录制时长
    try:
        duration_input = input("\n录制时长(秒, 默认30): ").strip()
        duration = 30 if duration_input == "" else int(duration_input)
        
        if duration <= 0:
            print("❌ 录制时长必须大于0")
            return
            
        print(f"✅ 录制时长: {duration} 秒")
        
    except (ValueError, KeyboardInterrupt):
        print("\n❌ 用户取消或输入无效")
        return
    
    # 开始录制
    print(f"\n{'='*60}")
    print("准备开始录制，按任意键继续或 Ctrl+C 取消...")
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ 用户取消")
        return
    
    try:
        output_file = record_to_mp3(device_index, channels, duration)
        
        if output_file:
            print(f"\n🎉 录制成功完成！")
            print(f"📁 文件保存位置: {output_file}")
            
            # 询问是否播放
            play_choice = input("\n是否播放录制的音频？(y/n, 默认n): ").strip().lower()
            if play_choice == 'y':
                try:
                    import os
                    if os.name == 'nt':  # Windows
                        os.startfile(output_file)
                    else:  # macOS/Linux
                        os.system(f"open '{output_file}'" if sys.platform == "darwin" else f"xdg-open '{output_file}'")
                    print("🔊 正在播放音频...")
                except Exception as e:
                    print(f"❌ 无法播放音频: {e}")
        else:
            print("❌ 录制失败")
            
    except KeyboardInterrupt:
        print("\n❌ 录制被用户中断")
    except Exception as e:
        print(f"❌ 录制过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()