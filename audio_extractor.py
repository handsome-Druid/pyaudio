# -*- coding: utf-8 -*-
"""
音频数据提取和分析工具 - 从JSON文件中提取完整的音频数据并进行分析或保存为音频文件
"""

import json
import numpy as np
import wave
from datetime import datetime
from pathlib import Path

def extract_audio_from_json(json_file, output_format='wav'):
    """
    从JSON文件中提取音频数据
    
    Args:
        json_file: JSON结果文件路径
        output_format: 输出格式，'wav' 或 'numpy'
        
    Returns:
        dict: 提取结果信息
    """
    try:
        print(f"📂 读取JSON文件: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        metadata = results['metadata']
        readings = results['readings']
        
        print("📋 文件信息:")
        print(f"   设备: {metadata['device_name']}")
        print(f"   通道数: {metadata['selected_channels']}")
        print(f"   采样率: {metadata['sample_rate']} Hz")
        print(f"   读取次数: {len(readings)}")
        print(f"   包含完整音频: {'是' if metadata.get('save_raw_audio', False) else '否'}")
        
        if not metadata.get('save_raw_audio', False):
            print("❌ 此文件不包含完整的音频数据")
            return None
        
        extracted_files = []
        
        # 处理每个读取结果
        for i, reading in enumerate(readings, 1):
            if 'raw_audio_data' not in reading:
                print(f"⚠️  第{i}次读取无音频数据，跳过")
                continue
            
            print(f"\n🎵 处理第 {i} 次读取的音频数据...")
            
            # 重建numpy数组
            raw_data = reading['raw_audio_data']
            if isinstance(raw_data, dict) and raw_data.get('_type') == 'numpy_array':
                audio_array = np.array(raw_data['data'], dtype=raw_data['dtype']).reshape(raw_data['shape'])
            else:
                print(f"❌ 第{i}次读取的音频数据格式不正确")
                continue
            
            channels, samples = audio_array.shape
            duration = samples / metadata['sample_rate']
            
            print(f"   数据形状: {audio_array.shape}")
            print(f"   时长: {duration:.2f} 秒")
            print(f"   数据范围: {audio_array.min():.6f} ~ {audio_array.max():.6f}")
            
            # 生成文件名
            read_time = datetime.fromisoformat(reading['read_time'].replace('Z', '+00:00'))
            timestamp = read_time.strftime("%Y%m%d_%H%M%S")
            base_name = f"extracted_audio_{timestamp}_read{i}"
            
            if output_format == 'wav':
                # 保存为WAV文件
                wav_filename = f"{base_name}.wav"
                
                with wave.open(wav_filename, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(metadata['sample_rate'])
                    
                    # 转换数据格式: (channels, samples) -> (samples, channels)
                    audio_interleaved = audio_array.T
                    # 转换为16-bit整数
                    audio_int16 = np.clip(audio_interleaved * 32767, -32768, 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
                
                file_size = Path(wav_filename).stat().st_size
                print(f"   💾 已保存WAV文件: {wav_filename} ({file_size:,} 字节)")
                extracted_files.append({
                    'type': 'wav',
                    'filename': wav_filename,
                    'size': file_size,
                    'duration': duration,
                    'channels': channels
                })
                
            elif output_format == 'numpy':
                # 保存为numpy文件
                npy_filename = f"{base_name}.npy"
                np.save(npy_filename, audio_array)
                
                file_size = Path(npy_filename).stat().st_size
                print(f"   💾 已保存NumPy文件: {npy_filename} ({file_size:,} 字节)")
                extracted_files.append({
                    'type': 'numpy',
                    'filename': npy_filename,
                    'size': file_size,
                    'duration': duration,
                    'channels': channels
                })
        
        return {
            'metadata': metadata,
            'extracted_files': extracted_files,
            'total_files': len(extracted_files)
        }
        
    except Exception as e:
        print(f"❌ 提取音频数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_extracted_audio(json_file):
    """
    分析JSON文件中的音频数据（不保存音频文件，仅分析）
    
    Args:
        json_file: JSON结果文件路径
    """
    try:
        print(f"🔍 分析JSON文件中的音频数据: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        metadata = results['metadata']
        readings = results['readings']
        
        print("\n📊 详细分析结果:")
        print(f"{'='*60}")
        
        all_rms_values = []
        all_peak_values = []
        total_duration = 0
        
        for i, reading in enumerate(readings, 1):
            if 'raw_audio_data' not in reading:
                continue
            
            print(f"\n🎵 第 {i} 次读取分析:")
            
            # 重建numpy数组
            raw_data = reading['raw_audio_data']
            if isinstance(raw_data, dict) and raw_data.get('_type') == 'numpy_array':
                audio_array = np.array(raw_data['data'], dtype=raw_data['dtype']).reshape(raw_data['shape'])
            else:
                continue
            
            channels, samples = audio_array.shape
            duration = samples / metadata['sample_rate']
            total_duration += duration
            
            # 读取时间
            read_time = datetime.fromisoformat(reading['read_time'].replace('Z', '+00:00'))
            print(f"   读取时间: {read_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   持续时间: {duration:.2f} 秒")
            
            # 分析每个通道
            for ch in range(channels):
                channel_data = audio_array[ch]
                
                # 基本统计
                rms = np.sqrt(np.mean(channel_data**2))
                peak = np.max(np.abs(channel_data))
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                
                # 转换为dB
                rms_db = 20 * np.log10(rms + 1e-10)
                peak_db = 20 * np.log10(peak + 1e-10)
                
                # 零交叉率
                zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
                zcr = zero_crossings / len(channel_data)
                
                # 频谱分析（简单）
                fft = np.fft.fft(channel_data)
                magnitude = np.abs(fft)[:len(fft)//2]
                dominant_freq_idx = np.argmax(magnitude)
                dominant_freq = dominant_freq_idx * metadata['sample_rate'] / len(fft)
                
                print(f"   通道 {ch}:")
                print(f"      RMS: {rms:.6f} ({rms_db:6.1f} dB)")
                print(f"      峰值: {peak:.6f} ({peak_db:6.1f} dB)")
                print(f"      均值: {mean:8.6f}")
                print(f"      标准差: {std:.6f}")
                print(f"      零交叉率: {zcr:.4f}")
                print(f"      主要频率: {dominant_freq:.1f} Hz")
                
                all_rms_values.append(rms_db)
                all_peak_values.append(peak_db)
        
        # 总体统计
        if all_rms_values:
            print("\n📈 总体统计:")
            print(f"   总音频时长: {total_duration:.2f} 秒")
            print(f"   平均RMS电平: {np.mean(all_rms_values):6.1f} dB")
            print(f"   RMS电平范围: {np.min(all_rms_values):6.1f} ~ {np.max(all_rms_values):6.1f} dB")
            print(f"   平均峰值电平: {np.mean(all_peak_values):6.1f} dB")
            print(f"   峰值电平范围: {np.min(all_peak_values):6.1f} ~ {np.max(all_peak_values):6.1f} dB")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主程序"""
    print("="*60)
    print("🎵 音频数据提取和分析工具")
    print("="*60)
    
    try:
        # 获取JSON文件
        json_file = input("请输入JSON文件路径: ").strip().strip('"')
        
        if not Path(json_file).exists():
            print(f"❌ 文件不存在: {json_file}")
            return
        
        # 选择操作模式
        print("\n选择操作模式:")
        print("1. 提取为WAV音频文件")
        print("2. 提取为NumPy数组文件")
        print("3. 仅分析（不保存音频文件）")
        
        choice = input("请选择 (1-3, 默认3): ").strip()
        
        if choice == '1':
            print("\n🎵 提取为WAV音频文件...")
            result = extract_audio_from_json(json_file, 'wav')
        elif choice == '2':
            print("\n🔢 提取为NumPy数组文件...")
            result = extract_audio_from_json(json_file, 'numpy')
        else:
            print("\n🔍 分析模式...")
            analyze_extracted_audio(json_file)
            return
        
        if result:
            print("\n✅ 提取完成！")
            print(f"   总共提取: {result['total_files']} 个文件")
            
            total_size = sum(f['size'] for f in result['extracted_files'])
            total_duration = sum(f['duration'] for f in result['extracted_files'])
            
            print(f"   总文件大小: {total_size:,} 字节 ({total_size/1024/1024:.2f} MB)")
            print(f"   总音频时长: {total_duration:.2f} 秒")
            
            print("\n📁 提取的文件:")
            for file_info in result['extracted_files']:
                print(f"   - {file_info['filename']} "
                      f"({file_info['size']:,} 字节, "
                      f"{file_info['duration']:.2f}s, "
                      f"{file_info['channels']}ch)")
        
    except KeyboardInterrupt:
        print("\n❌ 用户取消操作")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()