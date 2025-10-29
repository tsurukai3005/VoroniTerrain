#!/usr/bin/env python3
"""地形のスケールと妥当性を分析"""
import numpy as np

# v9の統計
print("=== v9 地形スケール分析 ===\n")

# 基本情報
map_size = 100  # ピクセル
height_min = 31.0
height_max = 77.8
height_diff = 46.8

print(f"マップサイズ: {map_size}x{map_size} ピクセル")
print(f"高度範囲: {height_min:.1f}m - {height_max:.1f}m")
print(f"高度差: {height_diff:.1f}m\n")

# 想定されるスケール
print("=== スケール想定 ===")
# 1ピクセル = 1mと仮定
pixel_scale = 1.0  # m/pixel
map_size_meters = map_size * pixel_scale
print(f"仮定: 1ピクセル = {pixel_scale}m")
print(f"マップ実寸: {map_size_meters}m x {map_size_meters}m = {map_size_meters/1000:.1f}km x {map_size_meters/1000:.1f}km\n")

# 傾斜角度の計算
print("=== 傾斜特性 ===")
# 崖: 8ピクセルで高度変化
cliff_distance = 8.0  # pixels
cliff_height = height_diff * 0.6  # 高度差の60%程度と仮定
cliff_angle = np.arctan(cliff_height / cliff_distance) * 180 / np.pi
print(f"崖の特性:")
print(f"  距離: {cliff_distance}m")
print(f"  高度変化: ~{cliff_height:.1f}m")
print(f"  角度: ~{cliff_angle:.1f}度\n")

# 坂: 30ピクセルで高度変化
slope_distance = 30.0  # pixels
slope_height = height_diff * 0.4  # 高度差の40%程度と仮定
slope_angle = np.arctan(slope_height / slope_distance) * 180 / np.pi
print(f"坂の特性:")
print(f"  距離: {slope_distance}m")
print(f"  高度変化: ~{slope_height:.1f}m")
print(f"  角度: ~{slope_angle:.1f}度\n")

# 現実の地形との比較
print("=== 現実の地形との比較 ===")
print("一般的な地形の傾斜角度:")
print("  平地: 0-5度")
print("  緩やかな丘: 5-15度")
print("  急な丘: 15-30度")
print("  険しい山: 30-45度")
print("  崖: 45-70度")
print("  垂直な崖: 70-90度\n")

print(f"v9の崖（{cliff_angle:.1f}度）: ", end="")
if cliff_angle < 45:
    print("険しい山程度 → やや緩い")
elif cliff_angle < 70:
    print("崖として適切 ✅")
else:
    print("垂直に近い → やや急すぎる")

print(f"v9の坂（{slope_angle:.1f}度）: ", end="")
if slope_angle < 15:
    print("緩やかな丘 ✅")
elif slope_angle < 30:
    print("急な丘 ✅")
else:
    print("険しい山 → やや急すぎる")

print("\n=== 推奨調整 ===")
# 理想的な崖と坂の角度
target_cliff_angle = 60  # 度
target_slope_angle = 20  # 度

# 必要な調整
cliff_height_target = cliff_distance * np.tan(target_cliff_angle * np.pi / 180)
slope_height_target = slope_distance * np.tan(target_slope_angle * np.pi / 180)

print(f"崖を{target_cliff_angle}度にするには:")
print(f"  高度変化: {cliff_height_target:.1f}m必要（現在~{cliff_height:.1f}m）")
if cliff_height_target > height_diff:
    print(f"  → 高度差を{cliff_height_target:.1f}m以上に増やす")
else:
    print(f"  → 現在の高度差で十分")

print(f"\n坂を{target_slope_angle}度にするには:")
print(f"  高度変化: {slope_height_target:.1f}m必要（現在~{slope_height:.1f}m）")
if slope_height_target > height_diff * 0.5:
    print(f"  → 高度差またはslope_rangeの調整が必要")

print("\n=== パラメータ推奨値 ===")
# 現在の高度差（46.8m）を前提
print(f"現在の高度差: {height_diff:.1f}m を前提として:")
print(f"  cliff_range: 8m (現状維持)")
print(f"  slope_range: 30m (現状維持)")
print(f"  cliff_power: 0.2 (急峻、現状維持)")
print(f"  slope_power: 0.7 (緩やか、現状維持)")

# より険しい地形にしたい場合
print(f"\nより険しい地形にする場合:")
print(f"  最低高度: 40m → 20m")
print(f"  最高高度: 95m (上限)")
print(f"  高度差: 75m（現在の1.6倍）")

# より緩やかな地形にしたい場合
print(f"\nより緩やかな地形にする場合:")
print(f"  cliff_range: 8m → 12m（崖の影響範囲拡大）")
print(f"  slope_range: 30m → 40m（坂の影響範囲拡大）")
print(f"  高度差: 30m（現在の0.64倍）")
