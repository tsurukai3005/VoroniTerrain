#!/usr/bin/env python3
"""区分境界の不連続を修正v2"""

# ファイルを読み込み
with open('terrain_voxel_v8_fixed.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 499行目の後に補間コードを追加
# primary_height計算の直後に補間ロジックを挿入

new_code = '''
                # === ステップ3: 区分境界での滑らかな補間（高度逆転防止版） ===
                if len(boundary_info) > 1:
                    # 2番目に方向が近い境界
                    secondary_boundary = boundary_info[1]

                    # primary境界の方向ベクトル
                    primary_key = primary_boundary['key']
                    primary_bc = self.boundary_centroids[primary_key]
                    primary_vec = np.array([primary_bc[0] - centroid[0], primary_bc[1] - centroid[1]])
                    primary_vec_norm = np.linalg.norm(primary_vec)

                    # secondary境界の方向ベクトル
                    secondary_key = secondary_boundary['key']
                    secondary_bc = self.boundary_centroids[secondary_key]
                    secondary_vec = np.array([secondary_bc[0] - centroid[0], secondary_bc[1] - centroid[1]])
                    secondary_vec_norm = np.linalg.norm(secondary_vec)

                    if primary_vec_norm > 0.1 and secondary_vec_norm > 0.1:
                        # 2つの境界方向の角度差
                        cos_boundaries = np.dot(primary_vec, secondary_vec) / (primary_vec_norm * secondary_vec_norm)
                        boundaries_angle = np.arccos(np.clip(cos_boundaries, -1.0, 1.0))

                        # pixel方向が2つの境界の「間」にあるかチェック
                        cross_ps = primary_vec[0] * secondary_vec[1] - primary_vec[1] * secondary_vec[0]
                        cross_pp = primary_vec[0] * pixel_vec[1] - primary_vec[1] * pixel_vec[0]
                        cross_xs = pixel_vec[0] * secondary_vec[1] - pixel_vec[1] * secondary_vec[0]

                        in_between = (np.sign(cross_ps) == np.sign(cross_pp) and np.sign(cross_ps) == np.sign(cross_xs))
                        almost_zero = abs(cross_ps) < 0.1 or abs(cross_pp) < 0.1 or abs(cross_xs) < 0.1

                        if in_between or almost_zero:
                            # 補間重み計算
                            cos_pixel_primary = np.dot(pixel_vec, primary_vec) / (pixel_norm * primary_vec_norm)
                            angle_pixel_primary = np.arccos(np.clip(cos_pixel_primary, -1.0, 1.0))

                            if boundaries_angle > 0.01:
                                interp_weight = np.clip(angle_pixel_primary / boundaries_angle, 0.0, 1.0)

                                # secondary境界の高度計算（高度逆転防止）
                                secondary_neighbor_height = self.group_heights[secondary_boundary['neighbor_id']]
                                secondary_distance = secondary_boundary['distance']
                                secondary_is_wall = self.is_wall.get(secondary_key, True)

                                if secondary_is_wall:
                                    if secondary_distance < 8.0:
                                        t = secondary_distance / 8.0
                                        sec_blend = 1.0 - (t ** 0.2)
                                    else:
                                        sec_blend = 0.0
                                    if base_height > secondary_neighbor_height:
                                        secondary_height = base_height * (1.0 - sec_blend) + secondary_neighbor_height * sec_blend
                                    else:
                                        secondary_height = base_height
                                else:
                                    if secondary_distance < 30.0:
                                        t = secondary_distance / 30.0
                                        sec_blend = 1.0 - (t ** 0.7)
                                    else:
                                        sec_blend = 0.0
                                    sec_boundary_height = (base_height + secondary_neighbor_height) / 2.0
                                    secondary_height = base_height * (1.0 - sec_blend) + sec_boundary_height * sec_blend

                                # 補間（base_heightから遠ざからない範囲で）
                                dist_primary = abs(primary_height - base_height)
                                dist_secondary = abs(secondary_height - base_height)

                                if dist_secondary < dist_primary * 1.5:
                                    smooth_weight = interp_weight * 0.3
                                    final_height = primary_height * (1.0 - smooth_weight) + secondary_height * smooth_weight
                                else:
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

# 499行目の後に挿入（0-indexed: 498の後）
result_lines = lines[:498] + [new_code] + lines[498:]

# ファイルに書き込み
with open('terrain_voxel_v9_smooth.py', 'w', encoding='utf-8') as f:
    f.writelines(result_lines)

print(f"✅ 区分境界滑らか版作成: terrain_voxel_v9_smooth.py")
