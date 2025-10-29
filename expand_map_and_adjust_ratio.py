#!/usr/bin/env python3
"""マップを4倍に拡大し、崖と坂の比率を調整"""

with open('terrain_voxel_v11_smooth_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. マップサイズを4倍に（100x100 → 200x200）
content = content.replace('width=100', 'width=200')
content = content.replace('height=100', 'height=200')
content = content.replace('self.width = 100', 'self.width = 200')
content = content.replace('self.height = 100', 'self.height = 200')

# 2. 崖と坂の比率を調整
# 現在: 通路比率 66% → 目標: 通路比率 55%（崖を増やす）
# 追加の通路を減らす: 40% → 25%
content = content.replace(
    'num_additional_passages = int(len(all_walls) * 0.4)',
    'num_additional_passages = int(len(all_walls) * 0.25)'
)

# 3. 壁の最低高度差を調整（緩やかすぎないように）
# 12m → 15m（崖をより明確に）
content = content.replace('min_diff = 12', 'min_diff = 15')

# 4. 領域数を調整（マップが4倍なので適切に増やす）
# セル数: 200 → 350（約1.75倍、密度を少し下げる）
# グループ数: 15 → 25（約1.67倍）
content = content.replace('num_cells=200', 'num_cells=350')
content = content.replace('num_groups=15', 'num_groups=25')

# 5. 崖と坂の範囲もスケールに合わせて調整
# cliff_range: 12m → 15m（マップが広くなったので範囲も拡大）
# slope_range: 45m → 60m（同様に拡大）
content = content.replace('cliff_range = 12.0', 'cliff_range = 15.0')
content = content.replace('slope_range = 45.0', 'slope_range = 60.0')

with open('terrain_voxel_v12_expanded.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ マップ拡大＆比率調整版作成: terrain_voxel_v12_expanded.py")
print("\n変更内容:")
print("  1. マップサイズ: 100x100 → 200x200（4倍）")
print("  2. セル数: 200 → 350")
print("  3. グループ数: 15 → 25")
print("  4. 追加通路割合: 40% → 25%（崖を増やす）")
print("  5. 壁最低高度差: 12m → 15m（崖を明確に）")
print("  6. cliff_range: 12m → 15m")
print("  7. slope_range: 45m → 60m")
print("\n期待される効果:")
print("  - より広い空間でなだらかな地形を実現")
print("  - 崖の比率: 34% → 約42%")
print("  - 坂の比率: 66% → 約58%")
