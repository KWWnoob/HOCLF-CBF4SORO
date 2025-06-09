import csv
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path("/Users/sheepydoggy/Documents/GitHub/safety-aware-soft-robot-control")

# 文件名和图例
files = {
    'metrics_alpha_-20.csv': 'No CBF',
    'metrics_alpha_2.csv': 'Unsafe Zone Avoidance CBF',
    'metrics_alpha_10.csv': 'Contact Force CBF'
}

# Step 1: 收集所有 h 值来找全局 min/max
all_hs = []
for filename in files:
    filepath = base_dir / filename
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_hs.append(float(row['h']))

global_min = min(all_hs)
global_max = max(all_hs)

# Step 2: 画图时用全局 min/max 做归一化
plt.figure(figsize=(6, 3))
for filename, label in files.items():
    filepath = base_dir / filename
    times, hs = [], []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            h = float(row['h'])
            hs.append((h - global_min) / (global_max - global_min + 1e-8))  # 用全局归一化
    plt.plot(times, hs, label=label, linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Normalized CBF value')
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
