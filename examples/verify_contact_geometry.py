import jax.numpy as jnp
from jax.numpy.linalg import svd, qr
import matplotlib.pyplot as plt

# Step 0: 你提供的原始 B 矩陣
B = jnp.array([
    [ 2.82574926e-04, -8.86464824e-04,  3.78924318e-04,  5.19823328e-05, -4.38618638e-04,  2.40064210e-04],
    [-8.86464824e-04,  3.93379672e-03,  0.00000000e+00, -1.35075375e-04,  1.46666396e-03, -1.59509080e-04],
    [ 3.78924318e-04,  0.00000000e+00,  3.93379672e-03,  4.68178736e-05,  1.59509080e-04,  1.46666396e-03],
    [ 5.19823328e-05, -1.35075375e-04,  4.68178736e-05,  1.25711318e-05, -9.67492163e-05,  4.62961100e-05],
    [-4.38618638e-04,  1.46666396e-03,  1.59509080e-04, -9.67492163e-05,  9.83864879e-04,  0.00000000e+00],
    [ 2.40064210e-04, -1.59509080e-04,  1.46666396e-03,  4.62961100e-05,  0.00000000e+00,  9.83864879e-04]
])

# Step 1: 使用 QR 分解作為變換矩陣
Q, R = qr(B)
T_qr = Q  # 正交變換矩陣 (6x6)

# Step 2: 變換後的新 B 矩陣（近似解耦輸入矩陣）
B_new = B @ T_qr

# Step 3: 可視化分析 B_new 結構（是否近似塊對角）
plt.figure(figsize=(6,5))
plt.imshow(jnp.abs(B_new), cmap='viridis')
plt.title("Abs(B @ T_qr): 解耦後的 B")
plt.colorbar(label='|value|')
plt.xlabel("T_qr Columns (ũ input space)")
plt.ylabel("B Rows (system output space)")
plt.grid(False)
plt.tight_layout()
plt.show()

# Step 4: 可選 - 輸出前 3 列主要集中在哪些行（解耦評估）
abs_B_new = jnp.abs(B_new)
segment1_rows = abs_B_new[:3, :3]
segment2_rows = abs_B_new[3:, 3:]

print(B_new)
print("Segment 1 block (rows 0–2, cols 0–2):")
print(segment1_rows)

print("\nSegment 2 block (rows 3–5, cols 3–5):")
print(segment2_rows)
