"""按 V0-V10 默认配置和 V11 smoke 顺序运行，记录 CPU 墙钟时间。"""

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
scripts = []
for number in range(12):
    pattern = f"capstone/V{number}-*.py" if number == 11 else f"course/*/code/V{number}-*.py"
    matches = list(ROOT.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"V{number}: expected one script, found {len(matches)}")
    scripts.append(matches[0])

started = time.perf_counter()
durations = []
for script in scripts:
    print(f"\n=== {script.name} ===", flush=True)
    command = [sys.executable, str(script)]
    if script.name.startswith("V11-"):
        command.append("--smoke-test")
    script_started = time.perf_counter()
    subprocess.run(command, cwd=ROOT.parent, check=True, timeout=120)
    duration = time.perf_counter() - script_started
    durations.append((script.name, duration))
    print(f"耗时={duration:.2f} 秒", flush=True)

total = time.perf_counter() - started
print("\nCPU 耗时汇总")
for name, duration in durations:
    print(f"{name}：{duration:.2f} 秒")
print(f"通过：V0-V10 默认测试及 V11 冒烟测试，共 {len(scripts)} 个代码快照，总耗时 {total:.2f} 秒")
