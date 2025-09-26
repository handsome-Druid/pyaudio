# -*- coding: utf-8 -*-
"""
严格读取最后4秒音频的脚本
- 使用严格接口 read_queue_by_time() 仅获取最后4秒数据
- 每次读取后等待4秒再读
- 支持通过参数指定循环次数、设备索引、通道数

用法示例：
    python strict_last4_reader.py --iterations 3 --device-index 0 --channels 2
"""

import argparse
import time
from datetime import datetime
import numpy as np
import wave
from pathlib import Path

from audio_interface import MultiMicAudioInterface


def db_from_rms(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-10)))


essay = """
说明：
- 本脚本使用严格的按时间戳读取接口 read_queue_by_time(start_ts, end_ts)。
- 如果缓存不足4秒（例如启动初期），严格接口会报错；脚本会等待直至足够数据后再开始循环。
- 每次循环：
  1) 获取当前缓存时间范围 (oldest, newest)
  2) 使用 [newest-4.0, newest] 作为严格读取区间
  3) 打印本次读取的基本统计（形状、时长、RMS等）
  4) 等待4秒进入下一轮
"""


def parse_args():
    p = argparse.ArgumentParser(description="严格读取最后4秒音频数据")
    p.add_argument("--iterations", type=int, default=3, help="循环次数，默认3")
    p.add_argument("--device-index", type=int, default=None, help="音频设备索引，默认None=系统默认设备")
    p.add_argument("--channels", type=int, default=2, help="通道数，默认2")
    p.add_argument("--sample-rate", type=int, default=44100, help="采样率，默认44100")
    p.add_argument("--chunk-size", type=int, default=1024, help="块大小（帧数），默认1024")
    p.add_argument("--sample-format", type=str, default="auto", choices=["auto", "int24", "int16", "float32"], help="采样格式，默认auto优先尝试24位")
    p.add_argument("--output", type=str, default=None, help="输出WAV文件名；未指定时自动生成")
    return p.parse_args()


def wait_until_enough_buffer(audio: MultiMicAudioInterface, need_seconds: float = 4.0, timeout: float = 15.0) -> bool:
    """等待缓存达到指定秒数；超时返回False"""
    deadline = time.time() + timeout
    last_report = 0
    while time.time() < deadline:
        oldest, newest = audio.get_queue_time_range()
        if oldest is not None and newest is not None:
            if (newest - oldest) >= need_seconds:
                return True
            # 每500ms报告一次当前进度
            now = time.time()
            if now - last_report > 0.5:
                have = newest - oldest
                print(f"🕒 等待缓存填充：当前≈{have:.2f}s / 目标 {need_seconds:.2f}s", end="\r", flush=True)
                last_report = now
        time.sleep(0.05)
    return False


def strict_read_last4(audio: MultiMicAudioInterface):
    """严格读取最后4秒，返回 (data, t0, t1)；若因竞争失败，尝试小次数快速重试"""
    # 小重试可以缓解 newest 变化导致的边界竞争
    for _ in range(3):
        oldest, newest = audio.get_queue_time_range()
        if oldest is None or newest is None:
            raise ValueError("缓存为空")
        start_ts = newest - 4.0
        end_ts = newest
        try:
            data, t0, t1 = audio.read_queue_by_time(start_ts, end_ts)
            return data, t0, t1
        except ValueError:
            time.sleep(0.02)
    # 最终失败则抛出
    data, t0, t1 = audio.read_queue_by_time(newest - 4.0, newest)
    return data, t0, t1


def main():
    args = parse_args()
    print("=" * 60)
    print("🎯 严格读取最后4秒音频")
    print("=" * 60)
    print(essay)

    with MultiMicAudioInterface(
        device_index=args.device_index,
        sample_rate=args.sample_rate,
        channels=args.channels,
        chunk_size=args.chunk_size,
        sample_format=args.sample_format,
    ) as audio:
        audio.start_recording()
        print("🎙️  已开始录音。")

        # 启动期：等待缓存至少达到4秒
        print("⏳ 正在等待缓存填充至 4.0 秒……")
        if not wait_until_enough_buffer(audio, need_seconds=4.0, timeout=30.0):
            print("❌ 在超时时间内缓存不足4秒，退出。")
            return
        print("\n✅ 缓存满足条件，开始严格读取循环。\n")

        collected = []  # 收集每次读取到的4秒数据 (channels, samples)
        time_ranges = []  # 收集每次的时间范围 (t0, t1)

        for i in range(1, args.iterations + 1):
            print(f"—— 第 {i}/{args.iterations} 次 ——")
            try:
                data, t0, t1 = strict_read_last4(audio)
            except ValueError as e:
                print(f"❌ 读取失败: {e}")
                break

            # 基本信息
            duration = data.shape[1] / audio.sample_rate
            print(f"时间范围: {datetime.fromtimestamp(t0).strftime('%H:%M:%S.%f')[:-3]} ~ {datetime.fromtimestamp(t1).strftime('%H:%M:%S.%f')[:-3]}")
            print(f"数据形状: {tuple(data.shape)} | 实际时长: {duration:.2f}s")

            # 简单电平分析
            for ch in range(data.shape[0]):
                rms = float(np.sqrt(np.mean(data[ch] ** 2)))
                print(f"  通道{ch}: RMS={db_from_rms(rms):6.1f} dB")

            # 保存本次结果以便最终拼接
            collected.append(data)
            time_ranges.append((t0, t1))

            if i < args.iterations:
                print("⏳ 等待 4 秒进入下一轮……\n")
                time.sleep(4.0)

        # 循环结束后，将所有片段拼接并写入WAV
        if collected:
            full = np.concatenate(collected, axis=1)  # (channels, total_samples)

            # 生成文件名
            if args.output:
                out_path = Path(args.output)
            else:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"strict_last4_concat_{ts}_sr{audio.sample_rate}_ch{audio.channels}_n{len(collected)}.wav"
                out_path = Path(out_name)

            # 写WAV（根据实际输入格式选择位深；支持int24/ int16）
            with wave.open(str(out_path), 'wb') as wf:
                wf.setnchannels(audio.channels)
                wf.setframerate(audio.sample_rate)
                interleaved_f = np.clip(full.T, -1.0, 1.0)
                if getattr(audio, "sample_format_name", "int16") == "int24":
                    # 24位写入
                    wf.setsampwidth(3)
                    # float32 -> int24 little-endian bytes
                    x = np.round(np.clip(interleaved_f * 8388608.0, -8388608, 8388607)).astype(np.int32)
                    b0 = (x & 0xFF).astype(np.uint8)
                    b1 = ((x >> 8) & 0xFF).astype(np.uint8)
                    b2 = ((x >> 16) & 0xFF).astype(np.uint8)
                    packed = np.stack([b0, b1, b2], axis=-1).reshape(-1)
                    wf.writeframes(packed.tobytes())
                else:
                    # 默认16位
                    wf.setsampwidth(2)
                    interleaved_i16 = (interleaved_f * 32767.0).astype(np.int16)  # (samples, channels)
                    wf.writeframes(interleaved_i16.tobytes())

            total_sec = full.shape[1] / audio.sample_rate
            tr0, tr1 = time_ranges[0][0], time_ranges[-1][1]
            print("\n💾 已保存拼接WAV:")
            print(f"  文件: {out_path.resolve()}")
            print(f"  总时长: {total_sec:.2f}s | 片段数: {len(collected)}")
            print(f"  覆盖时间: {datetime.fromtimestamp(tr0).strftime('%H:%M:%S.%f')[:-3]} ~ {datetime.fromtimestamp(tr1).strftime('%H:%M:%S.%f')[:-3]}")
        else:
            print("\n⚠️ 未收集到任何数据，未生成WAV文件。")

        print("\n🎉 完成。")


if __name__ == "__main__":
    main()
