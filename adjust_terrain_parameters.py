#!/usr/bin/env python3
"""地形パラメータを調整してより自然な地形に"""

with open('terrain_voxel_v9_smooth.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 調整1: 崖の範囲を拡大（8m → 12m）
content = content.replace('cliff_range = 8.0', 'cliff_range = 12.0')

# 調整2: 坂の範囲を拡大（30m → 45m）
content = content.replace('slope_range = 30.0', 'slope_range = 45.0')

# 調整3: 崖のpower値を調整（0.2 → 0.3でやや緩やかに）
content = content.replace('t ** 0.2', 't ** 0.3')

# 調整4: 坂のpower値を調整（0.7 → 0.8でより線形に）
content = content.replace('t ** 0.7', 't ** 0.8')

# 調整5: 高度の初期範囲を調整（40-60 → 35-55でやや狭く）
content = content.replace('heights[0] = 40 + random.random() * 20',
                         'heights[0] = 35 + random.random() * 20')

# 調整6: 通路の高度差を縮小（2-4m → 1.5-3m）
content = content.replace('height_diff = 2 + random.random() * 2',
                         'height_diff = 1.5 + random.random() * 1.5')

# 調整7: 壁の最低高度差を縮小（15m → 12m）
content = content.replace('min_diff = 15', 'min_diff = 12')

with open('terrain_voxel_v10_adjusted.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ パラメータ調整版作成: terrain_voxel_v10_adjusted.py")
print("\n調整内容:")
print("  1. cliff_range: 8m → 12m（崖の範囲拡大）")
print("  2. slope_range: 30m → 45m（坂の範囲拡大）")
print("  3. cliff_power: 0.2 → 0.3（崖をやや緩やかに）")
print("  4. slope_power: 0.7 → 0.8（坂をより線形に）")
print("  5. 初期高度: 40-60m → 35-55m（範囲縮小）")
print("  6. 通路高度差: 2-4m → 1.5-3m（縮小）")
print("  7. 壁最低高度差: 15m → 12m（縮小）")
print("\n期待される効果:")
print("  - 崖: 74度 → 約60-65度")
print("  - 坂: 32度 → 約20-25度")
print("  - より自然ななだらかな地形")
