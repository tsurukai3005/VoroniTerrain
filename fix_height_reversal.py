#!/usr/bin/env python3
"""高度逆転問題を修正"""

# ファイルを読み込み
with open('terrain_voxel_v7_no_interp.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 466-488行目の高度計算ロジックを修正
new_code = '''                if is_wall:
                    # 崖: 境界から8ピクセル以内で急激に変化
                    cliff_range = 8.0
                    if distance < cliff_range:
                        t = distance / cliff_range
                        # べき乗0.2 = 急峻（境界でストンと落ちる）
                        blend = 1.0 - (t ** 0.2)
                    else:
                        # 崖の範囲外: ベース高度を保つ
                        blend = 0.0

                    # 壁: 高い側だけが低い方に向かって下がる
                    if base_height > neighbor_height:
                        # 高い側 → 低い方へ下がる
                        target_height = neighbor_height
                        primary_height = base_height * (1.0 - blend) + target_height * blend
                    else:
                        # 低い側 → 変化なし（上がらない）
                        primary_height = base_height
                else:
                    # 坂: 境界から30ピクセルまで緩やかに変化
                    slope_range = 30.0
                    if distance < slope_range:
                        t = distance / slope_range
                        # べき乗0.7 = 緩やか（線形に近い）
                        blend = 1.0 - (t ** 0.7)
                    else:
                        # 坂の範囲外
                        blend = 0.0

                    # 坂: 両側が境界の中間高度に向かう
                    boundary_height = (base_height + neighbor_height) / 2.0
                    primary_height = base_height * (1.0 - blend) + boundary_height * blend
'''

# 新しいコードを行リストに変換
new_lines = new_code.split('\n')
if new_lines and not new_lines[-1].strip():
    new_lines = new_lines[:-1]
new_lines = [line + '\n' for line in new_lines]

# 466-488行目を置き換え（0-indexed: 465-487）
result_lines = lines[:465] + new_lines + lines[488:]

# ファイルに書き込み
with open('terrain_voxel_v8_fixed.py', 'w', encoding='utf-8') as f:
    f.writelines(result_lines)

print(f"✅ 高度逆転修正版作成: terrain_voxel_v8_fixed.py")
print(f"   置き換え: 466-488行目（高度計算ロジック）")
print(f"   修正内容:")
print(f"     - 壁: 高い側のみ下がる、低い側は変化なし")
print(f"     - 坂: 両側が中間高度に向かう")
