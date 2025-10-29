#!/usr/bin/env python3
"""区分境界の不連続を修正（高度逆転を起こさない補間）"""

# ファイルを読み込み
with open('terrain_voxel_v8_fixed.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 490-494行目の補間無効化部分を、適切な補間に置き換え
new_code = '''                # === ステップ3: 区分境界での滑らかな補間（高度逆転防止版） ===
                if len(boundary_info) > 1:
                    # 2番目に方向が近い境界
                    secondary_boundary = boundary_info[1]

                    # primary境界の方向ベクトル（重心→境界中心）
                    primary_key = primary_boundary['key']
                    primary_bc = self.boundary_centroids[primary_key]
                    primary_vec = np.array([primary_bc[0] - centroid[0],
                                           primary_bc[1] - centroid[1]])
                    primary_vec_norm = np.linalg.norm(primary_vec)

                    # secondary境界の方向ベクトル（重心→境界中心）
                    secondary_key = secondary_boundary['key']
                    secondary_bc = self.boundary_centroids[secondary_key]
                    secondary_vec = np.array([secondary_bc[0] - centroid[0],
                                             secondary_bc[1] - centroid[1]])
                    secondary_vec_norm = np.linalg.norm(secondary_vec)

                    if primary_vec_norm > 0.1 and secondary_vec_norm > 0.1:
                        # 2つの境界方向の角度差を計算
                        cos_boundaries = np.dot(primary_vec, secondary_vec) / (primary_vec_norm * secondary_vec_norm)
                        cos_boundaries = np.clip(cos_boundaries, -1.0, 1.0)
                        boundaries_angle = np.arccos(cos_boundaries)

                        # pixel方向が2つの境界の「間」にあるかチェック
                        cross_primary_secondary = primary_vec[0] * secondary_vec[1] - primary_vec[1] * secondary_vec[0]
                        cross_primary_pixel = primary_vec[0] * pixel_vec[1] - primary_vec[1] * pixel_vec[0]
                        cross_pixel_secondary = pixel_vec[0] * secondary_vec[1] - pixel_vec[1] * secondary_vec[0]

                        sign_check = (np.sign(cross_primary_secondary) == np.sign(cross_primary_pixel) and
                                     np.sign(cross_primary_secondary) == np.sign(cross_pixel_secondary))
                        almost_zero = abs(cross_primary_secondary) < 0.1 or abs(cross_primary_pixel) < 0.1 or abs(cross_pixel_secondary) < 0.1

                        if sign_check or almost_zero:
                            # pixel方向からprimary方向への角度
                            cos_pixel_primary = np.dot(pixel_vec, primary_vec) / (pixel_norm * primary_vec_norm)
                            cos_pixel_primary = np.clip(cos_pixel_primary, -1.0, 1.0)
                            angle_pixel_primary = np.arccos(cos_pixel_primary)

                            # 補間重み（0.0 = 完全にprimary, 1.0 = 完全にsecondary）
                            if boundaries_angle > 0.01:
                                interpolation_weight = angle_pixel_primary / boundaries_angle
                                interpolation_weight = np.clip(interpolation_weight, 0.0, 1.0)

                                # secondary境界の高度を計算（高度逆転防止版）
                                secondary_is_wall = self.is_wall.get(secondary_key, True)
                                secondary_neighbor_height = self.group_heights[secondary_boundary['neighbor_id']]
                                secondary_distance = secondary_boundary['distance']

                                if secondary_is_wall:
                                    cliff_range = 8.0
                                    if secondary_distance < cliff_range:
                                        t = secondary_distance / cliff_range
                                        secondary_blend = 1.0 - (t ** 0.2)
                                    else:
                                        secondary_blend = 0.0

                                    # 壁: 高い側のみ下がる
                                    if base_height > secondary_neighbor_height:
                                        secondary_height = base_height * (1.0 - secondary_blend) + secondary_neighbor_height * secondary_blend
                                    else:
                                        secondary_height = base_height
                                else:
                                    slope_range = 30.0
                                    if secondary_distance < slope_range:
                                        t = secondary_distance / slope_range
                                        secondary_blend = 1.0 - (t ** 0.7)
                                    else:
                                        secondary_blend = 0.0

                                    # 坂: 中間高度に向かう
                                    secondary_boundary_height = (base_height + secondary_neighbor_height) / 2.0
                                    secondary_height = base_height * (1.0 - secondary_blend) + secondary_boundary_height * secondary_blend

                                # 区分境界での補間（高度逆転を起こさない範囲で）
                                # primary_heightとsecondary_heightの間で補間
                                # ただし、base_heightから遠ざかる方向には補間しない
                                if abs(secondary_height - base_height) < abs(primary_height - base_height):
                                    # secondary_heightの方がbase_heightに近い → 補間OK
                                    smooth_weight = interpolation_weight * 0.3
                                    final_height = primary_height * (1.0 - smooth_weight) + secondary_height * smooth_weight
                                elif abs(secondary_height - base_height) < abs(primary_height - base_height) * 1.5:
                                    # やや近い → 弱い補間
                                    smooth_weight = interpolation_weight * 0.15
                                    final_height = primary_height * (1.0 - smooth_weight) + secondary_height * smooth_weight
                                else:
                                    # 遠すぎる → 補間しない
                                    final_height = primary_height
                            else:
                                final_height = primary_height
                        else:
                            final_height = primary_height
                    else:
                        final_height = primary_height
                else:
                    final_height = primary_height

                height_map[y, x] = final_height
'''

# 新しいコードを行リストに変換
new_lines = new_code.split('\n')
if new_lines and not new_lines[-1].strip():
    new_lines = new_lines[:-1]
new_lines = [line + '\n' for line in new_lines]

# 490-494行目を置き換え（0-indexed: 489-493）
result_lines = lines[:489] + new_lines + lines[494:]

# ファイルに書き込み
with open('terrain_voxel_v9_smooth.py', 'w', encoding='utf-8') as f:
    f.writelines(result_lines)

print(f"✅ 区分境界滑らか版作成: terrain_voxel_v9_smooth.py")
print(f"   修正内容:")
print(f"     - 区分境界での適切な補間を追加")
print(f"     - 高度逆転を防ぐ条件付き補間")
print(f"     - base_heightから遠ざかる補間を防止")
