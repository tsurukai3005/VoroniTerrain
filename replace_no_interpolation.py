#!/usr/bin/env python3
"""補間を完全に無効化したバージョンを作成"""

# ファイルを読み込み
with open('terrain_voxel_v6.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 490行目から始まる補間部分を、primary_heightのみを使用するシンプル版に置き換え
new_code = '''                # === ステップ3: 補間を無効化（primary境界のみ使用） ===
                # 区分境界での補間を完全に無効化して高度逆転問題を確認
                final_height = primary_height

                height_map[y, x] = final_height
'''

# 新しいコードを行リストに変換
new_lines = new_code.split('\n')
if new_lines and not new_lines[-1].strip():
    new_lines = new_lines[:-1]
new_lines = [line + '\n' for line in new_lines]

# 490行目から577行目（補間部分全体）を置き換え
# v6での補間コードは490-577行目
result_lines = lines[:489] + new_lines + lines[577:]

# ファイルに書き込み
with open('terrain_voxel_v7_no_interp.py', 'w', encoding='utf-8') as f:
    f.writelines(result_lines)

print(f"✅ 補間無効化版作成: terrain_voxel_v7_no_interp.py")
print(f"   置き換え: 490-577行目 → {len(new_lines)}行（primary_heightのみ使用）")
