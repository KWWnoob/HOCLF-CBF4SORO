import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

base_dir = Path("/Users/sheepydoggy/Documents/GitHub/safety-aware-soft-robot-control")

def read_and_normalize_h(file_path):
    times, hs = [], []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            hs.append(float(row['h']))
    
    # Normalize hs to [0, 1]
    h_min = min(hs)
    h_max = max(hs)
    hs_norm = [(h) / (h_max+ 1e-8) for h in hs]
    
    return times, hs_norm
''''''
def read_and_normalize_force(file_path):
    times, hs = [], []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            hs.append(float(row['h']))
    
    # Normalize hs to [0, 1]
    h_min = min(hs)
    h_max = max(hs)
    hs_norm = [1-(h) / (h_max+ 1e-8) for h in hs]
    
    return times, hs_norm
''''''
def read_and_normalize_V(file_path):
    times, Vs = [], []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            Vs.append(float(row['V']))
    
    V_max = max(Vs)
    V_min = min(Vs)
    Vs_norm = [(v - V_min) / (V_max - V_min + 1e-8) for v in Vs]  # 独立归一化

    return times, Vs_norm

def get_global_V_range(file_paths):
    all_Vs = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_Vs.append(float(row['V']))
    V_max = max(all_Vs)
    V_min = min(all_Vs)
    return V_min, V_max

def read_and_global_normalize_V(file_path, V_min, V_max):
    times, Vs = [], []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            Vs.append(float(row['V']))
    
    denom = V_max - V_min + 1e-8
    Vs_norm = [(v - V_min) / denom for v in Vs]
    return times, Vs_norm
'''
'''
def get_global_unorm_range_strict(file_paths):
    all_norms = []
    u_keys = [f"u_{i}" for i in range(6)]
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                u_vals = [float(row[k]) for k in u_keys]
                norm = np.linalg.norm(u_vals)
                all_norms.append(norm)
    return min(all_norms), max(all_norms)
def read_and_normalize_unorm_curve_strict(file_path, norm_min, norm_max):
    times, norms = [], []
    u_keys = [f"u_{i}" for i in range(6)]
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            u_vals = [float(row[k]) for k in u_keys]
            norm = np.linalg.norm(u_vals)
            norms.append(norm)
    
    denom = norm_max - norm_min + 1e-8
    norms_normed = [(v - norm_min) / denom for v in norms]
    return times, norms_normed

# 文件名和图例
files = {
    'metrics_with_u_alpha_10.csv': 'Safety-Unaware Control',
    'metrics_with_u_alpha_-23.csv': 'Contact Avoidance Control',
    'metrics_with_u_alpha_-2.csv': 'Contact-Force Limit Control',
}


plt.figure(figsize=(6, 3))
for filename, label in files.items():
    filepath = base_dir / filename
    time, h_norm = read_and_normalize_h(filepath)
    plt.plot(time, h_norm, label=label, linewidth=2)

# 添加红色透明“危险区域”背景
plt.axhspan(0.0, 0.35, color='red', alpha=0.30, label='Unsafe Zone')

plt.xlabel('Time (s)')
plt.ylabel('Normalized HOCBF value')
plt.legend(fontsize=12, loc='best', framealpha=0.3)
plt.grid(True)
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig("normalized_CBF.pdf")  # 保存PDF
plt.show()

# 画图
plt.figure(figsize=(6, 3))
for filename, label in files.items():
    filepath = base_dir / filename
    time, h_norm = read_and_normalize_force(filepath)
    plt.plot(time, h_norm, label=label, linewidth=2)

plt.axhspan(0.65, 1.00, color='red', alpha=0.30, label='Unsafe Zone')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Contact Force value')
plt.legend(fontsize=12, loc='best', framealpha=0.3)
plt.grid(True)
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig("normalized_contact_force.pdf")  # 保存PDF
plt.show()


# === Step 1: 全局 min/max ===
V_min, V_max = get_global_V_range(files)

plt.figure(figsize=(6, 3))
for filename, label in files.items():
    filepath = base_dir / filename
    times, Vs_norm = read_and_global_normalize_V(filepath, V_min, V_max)
    plt.plot(times, Vs_norm, label=label, linewidth=2)

plt.xlabel('Time')
plt.ylabel('Normalized HOCLF value')
plt.legend(fontsize=12, loc='best', framealpha=0.3)
plt.grid(True)
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig("normalized_clf.pdf")  # 保存PDF
plt.show()

'''input'''
norm_min, norm_max = get_global_unorm_range_strict(files)

plt.figure(figsize=(6, 3))
for filename, label in files.items():
    filepath = base_dir / filename
    times, norms = read_and_normalize_unorm_curve_strict(filepath, norm_min, norm_max)
    plt.plot(times, norms, label=label, linewidth=2)

plt.xlabel('Time (s)')
plt.ylabel('Normalized $\\|u(t)\\|$')
plt.legend(fontsize=12, loc='best', framealpha=0.3)
plt.grid(True)
plt.tight_layout()
plt.rcParams['pdf.fonttype'] = 42
plt.savefig("normalized_u_norm_curve.pdf")
plt.show()
