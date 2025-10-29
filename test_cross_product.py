#!/usr/bin/env python3
"""外積による「間」判定のテスト"""
import numpy as np
import matplotlib.pyplot as plt

# テストケース：2つの境界方向とピクセル方向
# 重心を原点とする

# Case 1: pixelが2つの境界の間にある
primary_vec = np.array([1.0, 0.0])  # 0度方向
secondary_vec = np.array([0.0, 1.0])  # 90度方向
pixel_vec = np.array([1.0, 1.0])  # 45度方向（間にある）

# 外積計算
cross_ps = primary_vec[0] * secondary_vec[1] - primary_vec[1] * secondary_vec[0]
cross_pp = primary_vec[0] * pixel_vec[1] - primary_vec[1] * pixel_vec[0]
cross_xs = pixel_vec[0] * secondary_vec[1] - pixel_vec[1] * secondary_vec[0]

print("=== Case 1: pixelが間にある（45度） ===")
print(f"primary (0度): {primary_vec}")
print(f"secondary (90度): {secondary_vec}")
print(f"pixel (45度): {pixel_vec}")
print(f"cross(primary, secondary) = {cross_ps}")
print(f"cross(primary, pixel) = {cross_pp}")
print(f"cross(pixel, secondary) = {cross_xs}")
print(f"sign_check = {np.sign(cross_ps) == np.sign(cross_pp) and np.sign(cross_ps) == np.sign(cross_xs)}")

# Case 2: pixelが2つの境界の外にある
pixel_vec2 = np.array([1.0, -1.0])  # -45度方向（間にない）

cross_pp2 = primary_vec[0] * pixel_vec2[1] - primary_vec[1] * pixel_vec2[0]
cross_xs2 = pixel_vec2[0] * secondary_vec[1] - pixel_vec2[1] * secondary_vec[0]

print("\n=== Case 2: pixelが外にある（-45度） ===")
print(f"pixel (-45度): {pixel_vec2}")
print(f"cross(primary, pixel) = {cross_pp2}")
print(f"cross(pixel, secondary) = {cross_xs2}")
print(f"sign_check = {np.sign(cross_ps) == np.sign(cross_pp2) and np.sign(cross_ps) == np.sign(cross_xs2)}")

# Case 3: pixelが2つの境界の外（反対側）
pixel_vec3 = np.array([-1.0, -1.0])  # 225度方向（完全に外）

cross_pp3 = primary_vec[0] * pixel_vec3[1] - primary_vec[1] * pixel_vec3[0]
cross_xs3 = pixel_vec3[0] * secondary_vec[1] - pixel_vec3[1] * secondary_vec[0]

print("\n=== Case 3: pixelが完全に外（225度） ===")
print(f"pixel (225度): {pixel_vec3}")
print(f"cross(primary, pixel) = {cross_pp3}")
print(f"cross(pixel, secondary) = {cross_xs3}")
print(f"sign_check = {np.sign(cross_ps) == np.sign(cross_pp3) and np.sign(cross_ps) == np.sign(cross_xs3)}")

# 可視化
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.grid(True, alpha=0.3)

# 原点
ax.plot(0, 0, 'ko', markersize=10, label='Centroid')

# Primary境界方向（赤）
ax.arrow(0, 0, primary_vec[0], primary_vec[1], head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
ax.text(primary_vec[0]+0.1, primary_vec[1]+0.1, 'Primary (0°)', color='red', fontsize=12)

# Secondary境界方向（青）
ax.arrow(0, 0, secondary_vec[0], secondary_vec[1], head_width=0.1, head_length=0.1, fc='blue', ec='blue', linewidth=2)
ax.text(secondary_vec[0]+0.1, secondary_vec[1]+0.1, 'Secondary (90°)', color='blue', fontsize=12)

# Pixel方向（テストケース）
pixel_cases = [
    (pixel_vec, '45° (間)', 'green'),
    (pixel_vec2, '-45° (外)', 'orange'),
    (pixel_vec3, '225° (外)', 'purple')
]

for pv, label, color in pixel_cases:
    pv_norm = pv / np.linalg.norm(pv) * 1.5
    ax.arrow(0, 0, pv_norm[0], pv_norm[1], head_width=0.08, head_length=0.08, fc=color, ec=color, linewidth=1.5, alpha=0.7)
    ax.text(pv_norm[0]+0.2, pv_norm[1]+0.2, label, color=color, fontsize=10)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('外積による「間」判定のテスト', fontsize=14, fontweight='bold')
ax.set_aspect('equal')
ax.legend()

plt.tight_layout()
plt.savefig('cross_product_test.png', dpi=100)
print("\n✅ 可視化を保存: cross_product_test.png")
