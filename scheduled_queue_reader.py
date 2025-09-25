# -*- coding: utf-8 -*-
"""
音频队列定时读取脚本
每10秒运行一次队列读取（获取5秒音频数据），重复3次，保存结果到JSON
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from audio_interface import MultiMicAudioInterface
from device_detector import list_devices_simple

class NumpyJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持numpy数组和其他特殊类型"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                '_type': 'numpy_array',
                'data': obj.tolist(),
                'shape': obj.shape,
                'dtype': str(obj.dtype)
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def analyze_audio_data(audio_data, sample_rate):
    """
    分析音频数据，提取有用的统计信息
    
    Args:
        audio_data: 音频数据 (channels, samples)
        sample_rate: 采样率
        
    Returns:
        dict: 分析结果
    """
    if audio_data is None:
        return None
    
    channels, samples = audio_data.shape
    duration = samples / sample_rate
    
    analysis = {
        'shape': (channels, samples),
        'duration_seconds': float(duration),
        'sample_rate': sample_rate,
        'channels': []
    }
    
    # 分析每个通道
    for ch in range(channels):
        channel_data = audio_data[ch]
        
        # 计算统计信息
        rms = float(np.sqrt(np.mean(channel_data**2)))
        peak = float(np.max(np.abs(channel_data)))
        mean = float(np.mean(channel_data))
        std = float(np.std(channel_data))
        
        # 转换为dB
        rms_db = 20 * np.log10(rms + 1e-10)
        peak_db = 20 * np.log10(peak + 1e-10)
        
        # 计算零交叉率（Zero Crossing Rate）
        zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
        zcr = zero_crossings / len(channel_data)
        
        channel_info = {
            'channel_index': ch,
            'rms_linear': rms,
            'rms_db': float(rms_db),
            'peak_linear': peak,
            'peak_db': float(peak_db),
            'mean': mean,
            'std': std,
            'zero_crossing_rate': float(zcr),
            # 可选择保存原始数据（注意：会让JSON文件很大）
            # 'raw_data': channel_data  # 取消注释以保存原始音频数据
        }
        
        analysis['channels'].append(channel_info)
    
    return analysis

def scheduled_queue_reader(device_index=None, channels=None, output_file=None, interval=10, iterations=3, queue_duration=5.0, save_raw_audio=True):
    """
    定时队列读取器
    
    Args:
        device_index: 音频设备索引，None为默认设备
        channels: 通道数，None为自动选择
        output_file: 输出JSON文件路径
        interval: 读取间隔（秒）
        iterations: 重复次数
        queue_duration: 每次读取的队列时长（秒）
        save_raw_audio: 是否保存完整的音频数据
    """
    
    # 生成输出文件名
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"queue_reading_results_{timestamp}.json"
    
    print("="*60)
    print("🕒 音频队列定时读取器")
    print("="*60)
    print(f"📋 配置信息:")
    print(f"   读取间隔: {interval} 秒")
    print(f"   重复次数: {iterations} 次")
    print(f"   队列时长: {queue_duration} 秒")
    print(f"   保存完整音频数据: {'是' if save_raw_audio else '否'}")
    print(f"   输出文件: {output_file}")
    
    # 获取设备信息
    try:
        devices = list_devices_simple()
        if not devices:
            print("❌ 未找到可用的音频输入设备！")
            return None
        
        # 选择设备
        if device_index is None:
            device_idx, device_name, max_channels = devices[0]
        else:
            # 查找指定设备
            device_info = None
            for idx, name, max_ch in devices:
                if idx == device_index:
                    device_info = (idx, name, max_ch)
                    break
            
            if device_info is None:
                print(f"❌ 未找到设备索引 {device_index}，使用默认设备")
                device_idx, device_name, max_channels = devices[0]
            else:
                device_idx, device_name, max_channels = device_info
        
        # 确定通道数
        if channels is None:
            selected_channels = min(2, max_channels)  # 默认使用最多2通道
        else:
            if channels > max_channels:
                print(f"⚠️  请求的通道数 {channels} 超过设备最大支持 {max_channels}，使用最大支持通道数")
                selected_channels = max_channels
            else:
                selected_channels = channels
        
        print(f"✅ 使用设备: {device_name} (设备{device_idx})")
        print(f"✅ 通道配置: {selected_channels}/{max_channels} 通道")
        
    except Exception as e:
        print(f"❌ 设备检测失败: {e}")
        return None
    
    # 准备结果存储
    results = {
        'metadata': {
            'script_name': 'scheduled_queue_reader',
            'created_at': datetime.now().isoformat(),
            'device_index': device_idx,
            'device_name': device_name,
            'max_channels': max_channels,
            'selected_channels': selected_channels,
            'sample_rate': 44100,
            'chunk_size': 1024,
            'interval_seconds': interval,
            'iterations': iterations,
            'queue_duration_seconds': queue_duration,
            'save_raw_audio': save_raw_audio
        },
        'readings': []
    }
    
    # 创建音频接口
    try:
        with MultiMicAudioInterface(
            device_index=device_idx,
            channels=selected_channels,
            sample_rate=44100,
            chunk_size=1024
        ) as audio:
            
            print(f"\n🎙️  开始录音...")
            audio.start_recording()
            
            # 等待队列初始填充
            print("📦 等待队列初始填充（3秒）...")
            time.sleep(3)
            
            # 执行定时读取
            for iteration in range(1, iterations + 1):
                print(f"\n{'='*50}")
                print(f"📊 第 {iteration}/{iterations} 次读取")
                print(f"{'='*50}")
                
                # 记录读取开始时间
                read_start_time = time.time()
                read_start_datetime = datetime.now()
                
                # 获取队列状态
                queue_status = audio.get_queue_status()
                print(f"📈 队列状态:")
                print(f"   帧数: {queue_status['frame_count']}/{queue_status['max_frames']}")
                print(f"   时长: {queue_status['duration']:.2f}s")
                print(f"   状态: {'已满' if queue_status['is_full'] else '填充中'}")
                
                # 读取指定时长的队列数据
                print(f"📥 读取最近 {queue_duration} 秒的音频数据...")
                audio_data, start_timestamp, end_timestamp = audio.read_queue_duration(queue_duration)
                
                # 处理读取结果
                if audio_data is not None:
                    # 转换时间戳
                    start_datetime = datetime.fromtimestamp(start_timestamp)
                    end_datetime = datetime.fromtimestamp(end_timestamp)
                    actual_duration = audio_data.shape[1] / audio.sample_rate
                    
                    print(f"✅ 成功读取音频数据:")
                    print(f"   数据形状: {audio_data.shape}")
                    print(f"   实际时长: {actual_duration:.2f} 秒")
                    print(f"   时间范围: {start_datetime.strftime('%H:%M:%S.%f')[:-3]} ~ {end_datetime.strftime('%H:%M:%S.%f')[:-3]}")
                    
                    # 分析音频数据
                    print("🔍 分析音频数据...")
                    analysis = analyze_audio_data(audio_data, audio.sample_rate)
                    
                    # 显示分析结果
                    for ch_info in analysis['channels']:
                        ch = ch_info['channel_index']
                        rms_db = ch_info['rms_db']
                        peak_db = ch_info['peak_db']
                        zcr = ch_info['zero_crossing_rate']
                        print(f"   通道{ch}: RMS={rms_db:6.1f}dB, Peak={peak_db:6.1f}dB, ZCR={zcr:.4f}")
                    
                    # 保存到结果中
                    reading_result = {
                        'iteration': iteration,
                        'read_time': read_start_datetime.isoformat(),
                        'read_timestamp': read_start_time,
                        'queue_status': queue_status,
                        'audio_start_time': start_datetime.isoformat(),
                        'audio_end_time': end_datetime.isoformat(),
                        'audio_start_timestamp': start_timestamp,
                        'audio_end_timestamp': end_timestamp,
                        'requested_duration': queue_duration,
                        'actual_duration': actual_duration,
                        'analysis': analysis
                    }
                    
                    # 根据配置决定是否保存完整音频数据
                    if save_raw_audio:
                        reading_result['raw_audio_data'] = audio_data
                        print(f"💾 已保存完整音频数据到结果中")
                    
                    results['readings'].append(reading_result)
                    
                else:
                    print("❌ 未能读取到音频数据")
                    error_result = {
                        'iteration': iteration,
                        'read_time': read_start_datetime.isoformat(),
                        'read_timestamp': read_start_time,
                        'queue_status': queue_status,
                        'error': 'No audio data available'
                    }
                    results['readings'].append(error_result)
                
                # 等待下一次读取（除了最后一次）
                if iteration < iterations:
                    print(f"⏳ 等待 {interval} 秒后进行下一次读取...")
                    time.sleep(interval)
            
            print(f"\n{'='*60}")
            print("📊 所有读取完成！")
            
    except KeyboardInterrupt:
        print("\n❌ 用户中断操作")
        results['metadata']['interrupted'] = True
        results['metadata']['interrupt_time'] = datetime.now().isoformat()
    except Exception as e:
        print(f"❌ 录音过程中出错: {e}")
        import traceback
        traceback.print_exc()
        results['metadata']['error'] = str(e)
        results['metadata']['error_time'] = datetime.now().isoformat()
    
    # 保存结果到JSON文件
    try:
        print(f"\n💾 保存结果到: {output_file}")
        
        # 添加完成时间戳
        results['metadata']['completed_at'] = datetime.now().isoformat()
        results['metadata']['total_readings'] = len(results['readings'])
        
        # 保存JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
        
        # 显示文件统计信息
        file_path = Path(output_file)
        file_size = file_path.stat().st_size
        print(f"✅ 文件保存成功:")
        print(f"   文件路径: {file_path.absolute()}")
        print(f"   文件大小: {file_size:,} 字节 ({file_size/1024:.2f} KB)")
        print(f"   读取次数: {len(results['readings'])}")
        
        # 显示结果摘要
        successful_readings = [r for r in results['readings'] if 'analysis' in r]
        if successful_readings:
            print(f"\n📈 结果摘要:")
            print(f"   成功读取: {len(successful_readings)} 次")
            
            # 计算平均电平
            all_rms_db = []
            for reading in successful_readings:
                for ch_info in reading['analysis']['channels']:
                    all_rms_db.append(ch_info['rms_db'])
            
            if all_rms_db:
                avg_rms_db = sum(all_rms_db) / len(all_rms_db)
                min_rms_db = min(all_rms_db)
                max_rms_db = max(all_rms_db)
                print(f"   平均RMS电平: {avg_rms_db:.1f} dB")
                print(f"   RMS电平范围: {min_rms_db:.1f} ~ {max_rms_db:.1f} dB")
        
        return output_file
        
    except Exception as e:
        print(f"❌ 保存JSON文件失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_and_display_results(json_file):
    """
    加载并显示JSON结果文件
    
    Args:
        json_file: JSON文件路径
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("="*60)
        print("📊 队列读取结果分析")
        print("="*60)
        
        # 显示元数据
        metadata = results['metadata']
        print(f"📋 基本信息:")
        print(f"   创建时间: {metadata['created_at']}")
        print(f"   设备: {metadata['device_name']} (索引{metadata['device_index']})")
        print(f"   通道配置: {metadata['selected_channels']}/{metadata['max_channels']} 通道")
        print(f"   采样率: {metadata['sample_rate']} Hz")
        print(f"   读取间隔: {metadata['interval_seconds']} 秒")
        print(f"   保存完整音频: {'是' if metadata.get('save_raw_audio', False) else '否'}")
        print(f"   总读取次数: {metadata['total_readings']}")
        
        # 显示各次读取结果
        readings = results['readings']
        for i, reading in enumerate(readings, 1):
            print(f"\n📊 第 {i} 次读取:")
            print(f"   读取时间: {reading['read_time']}")
            
            if 'analysis' in reading:
                analysis = reading['analysis']
                print(f"   音频时长: {analysis['duration_seconds']:.2f} 秒")
                print(f"   数据形状: {tuple(analysis['shape'])}")
                
                for ch_info in analysis['channels']:
                    ch = ch_info['channel_index']
                    print(f"   通道{ch}: RMS={ch_info['rms_db']:6.1f}dB, Peak={ch_info['peak_db']:6.1f}dB")
                
                # 显示是否包含完整音频数据
                if 'raw_audio_data' in reading:
                    raw_data = reading['raw_audio_data']
                    if isinstance(raw_data, dict) and raw_data.get('_type') == 'numpy_array':
                        data_shape = tuple(raw_data['shape'])
                        data_size = len(str(raw_data['data']))  # 估算数据大小
                        print(f"   💾 完整音频数据: 形状{data_shape}, 数据大小约{data_size//1000}KB")
                    else:
                        print(f"   💾 包含完整音频数据")
                else:
                    print(f"   📊 仅包含分析结果（无完整音频数据）")
                    
            else:
                print(f"   ❌ 读取失败: {reading.get('error', '未知错误')}")
    
    except Exception as e:
        print(f"❌ 加载JSON文件失败: {e}")

def interactive_device_selection():
    """
    交互式设备和通道选择
    
    Returns:
        tuple: (device_index, channels) 或 (None, None) 如果用户取消
    """
    try:
        # 获取设备列表
        devices = list_devices_simple()
        if not devices:
            print("❌ 未找到可用的音频输入设备！")
            return None, None
        
        print(f"\n📱 发现 {len(devices)} 个可用设备，请选择:")
        print("-" * 60)
        
        for i, (device_idx, device_name, max_channels) in enumerate(devices, 1):
            print(f"{i:2d}. 设备 {device_idx:2d}: {device_name}")
            print(f"     最大通道数: {max_channels}")
            print()
        
        # 设备选择
        while True:
            try:
                choice = input(f"请选择设备 (1-{len(devices)}, 默认1): ").strip()
                
                if choice == "":
                    device_choice = 0
                else:
                    device_choice = int(choice) - 1
                
                if 0 <= device_choice < len(devices):
                    selected_device = devices[device_choice]
                    device_idx, device_name, max_channels = selected_device
                    break
                else:
                    print(f"❌ 请输入 1 到 {len(devices)} 之间的数字")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
        
        print(f"✅ 已选择: {device_name} (设备 {device_idx})")
        
        # 通道数选择
        if max_channels == 0:
            print("❌ 该设备不支持音频输入")
            return None, None
        
        print(f"\n🎵 该设备最大支持 {max_channels} 个输入通道")
        
        while True:
            try:
                channels_input = input(f"请输入通道数 (1-{max_channels}, 默认{min(2, max_channels)}): ").strip()
                
                if channels_input == "":
                    channels = min(2, max_channels)
                else:
                    channels = int(channels_input)
                
                if 1 <= channels <= max_channels:
                    break
                else:
                    print(f"❌ 通道数必须在 1-{max_channels} 之间")
                    
            except ValueError:
                print("❌ 请输入有效的数字")
        
        print(f"✅ 已选择 {channels} 个通道")
        
        return device_idx, channels
        
    except KeyboardInterrupt:
        print("\n❌ 用户取消选择")
        return None, None

def main():
    """主程序"""
    print("="*60)
    print("🕒 音频队列定时读取脚本")
    print("="*60)
    
    # 配置参数
    default_interval = 10      # 读取间隔（秒）
    default_iterations = 3     # 重复次数
    default_queue_duration = 5.0  # 每次读取的队列时长（秒）
    
    try:
        # 交互式配置
        print("📋 配置读取参数:")
        
        # 读取间隔
        try:
            interval_input = input(f"读取间隔(秒, 默认{default_interval}): ").strip()
            interval = default_interval if interval_input == "" else int(interval_input)
            if interval <= 0:
                interval = default_interval
                print(f"使用默认间隔: {interval} 秒")
        except ValueError:
            interval = default_interval
            print(f"输入无效，使用默认间隔: {interval} 秒")
        
        # 重复次数
        try:
            iterations_input = input(f"重复次数(默认{default_iterations}): ").strip()
            iterations = default_iterations if iterations_input == "" else int(iterations_input)
            if iterations <= 0:
                iterations = default_iterations
                print(f"使用默认次数: {iterations} 次")
        except ValueError:
            iterations = default_iterations
            print(f"输入无效，使用默认次数: {iterations} 次")
        
        # 队列读取时长
        try:
            duration_input = input(f"每次读取队列时长(秒, 默认{default_queue_duration}): ").strip()
            queue_duration = default_queue_duration if duration_input == "" else float(duration_input)
            if queue_duration <= 0 or queue_duration > 5.0:
                queue_duration = default_queue_duration
                print(f"使用默认时长: {queue_duration} 秒")
        except ValueError:
            queue_duration = default_queue_duration
            print(f"输入无效，使用默认时长: {queue_duration} 秒")
        
        # 是否保存完整音频数据
        save_raw_input = input("是否保存完整音频数据到JSON？(y/n, 默认y): ").strip().lower()
        save_raw_audio = save_raw_input != 'n'
        
        if save_raw_audio:
            print("⚠️  注意: 保存完整音频数据会使JSON文件变得很大")
        
        # 设备和通道选择
        device_index, channels = interactive_device_selection()
        if device_index is None:
            print("❌ 未选择设备，程序退出")
            return
        
        # 输出文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"queue_reading_results_{timestamp}.json"
        filename_input = input(f"输出文件名(默认: {default_filename}): ").strip()
        output_file = filename_input if filename_input else default_filename
        
        print(f"\n{'='*60}")
        print("📊 最终配置:")
        print(f"   读取间隔: {interval} 秒")
        print(f"   重复次数: {iterations} 次")
        print(f"   队列时长: {queue_duration} 秒")
        print(f"   设备索引: {device_index}")
        print(f"   通道数: {channels}")
        print(f"   保存完整音频: {'是' if save_raw_audio else '否'}")
        print(f"   输出文件: {output_file}")
        print(f"{'='*60}")
        
        # 确认开始
        confirm = input("\n按回车键开始，或输入 'q' 退出: ").strip().lower()
        if confirm == 'q':
            print("❌ 用户取消操作")
            return
        
        # 执行定时读取
        result_file = scheduled_queue_reader(
            device_index=device_index,
            channels=channels,
            output_file=output_file,
            interval=interval,
            iterations=iterations,
            queue_duration=queue_duration,
            save_raw_audio=save_raw_audio
        )
        
        if result_file:
            print(f"\n🎉 任务完成！结果已保存到: {result_file}")
            
            # 询问是否显示结果
            try:
                choice = input("\n是否显示结果摘要？(y/n, 默认y): ").strip().lower()
                if choice != 'n':
                    load_and_display_results(result_file)
            except KeyboardInterrupt:
                print("\n用户取消")
        else:
            print("❌ 任务执行失败")
            
    except KeyboardInterrupt:
        print("\n❌ 用户中断程序")
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()