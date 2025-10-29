#!/usr/bin/env python3
"""区分境界の完全な滑らか化 - 角度比による重み付け平均"""

with open('terrain_voxel_v10_adjusted.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# v10の498行目（primary_height計算後）に新しい補間ロジックを挿入
# 完全に書き直して、角度比による重み付け平均を実装

new_interpolation_code = '''
                # === ステップ3: 区分境界での完全な滑らかな補間 ===
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
                        # pixel方向が2つの境界の「間」にあるかチェック
                        cross_ps = primary_vec[0] * secondary_vec[1] - primary_vec[1] * secondary_vec[0]
                        cross_pp = primary_vec[0] * pixel_vec[1] - primary_vec[1] * pixel_vec[0]
                        cross_xs = pixel_vec[0] * secondary_vec[1] - pixel_vec[1] * secondary_vec[0]

                        in_between = (np.sign(cross_ps) == np.sign(cross_pp) and np.sign(cross_ps) == np.sign(cross_xs))
                        almost_zero = abs(cross_ps) < 0.1 or abs(cross_pp) < 0.1 or abs(cross_xs) < 0.1

                        if in_between or almost_zero:
                            # === 角度比による重み付け平均 ===
                            # pixel方向からprimary方向への角度
                            cos_pixel_primary = np.dot(pixel_vec, primary_vec) / (pixel_norm * primary_vec_norm)
                            angle_to_primary = np.arccos(np.clip(cos_pixel_primary, -1.0, 1.0))

                            # pixel方向からsecondary方向への角度
                            cos_pixel_secondary = np.dot(pixel_vec, secondary_vec) / (pixel_norm * secondary_vec_norm)
                            angle_to_secondary = np.arccos(np.clip(cos_pixel_secondary, -1.0, 1.0))

                            # 角度の合計（ゼロ除算回避）
                            total_angle = angle_to_primary + angle_to_secondary
                            if total_angle > 0.01:
                                # 重み計算：角度が近い方が重い
                                weight_primary = angle_to_secondary / total_angle
                                weight_secondary = angle_to_primary / total_angle

                                # secondary境界の高度計算（高度逆転防止版）
                                secondary_neighbor_height = self.group_heights[secondary_boundary['neighbor_id']]
                                secondary_distance = secondary_boundary['distance']
                                secondary_is_wall = self.is_wall.get(secondary_key, True)

                                if secondary_is_wall:
                                    if secondary_distance < 12.0:
                                        t = secondary_distance / 12.0
                                        sec_blend = 1.0 - (t ** 0.3)
                                    else:
                                        sec_blend = 0.0
                                    if base_height > secondary_neighbor_height:
                                        secondary_height = base_height * (1.0 - sec_blend) + secondary_neighbor_height * sec_blend
                                    else:
                                        secondary_height = base_height
                                else:
                                    if secondary_distance < 45.0:
                                        t = secondary_distance / 45.0
                                        sec_blend = 1.0 - (t ** 0.8)
                                    else:
                                        sec_blend = 0.0
                                    sec_boundary_height = (base_height + secondary_neighbor_height) / 2.0
                                    secondary_height = base_height * (1.0 - sec_blend) + sec_boundary_height * sec_blend

                                # 角度比による完全な重み付け平均
                                final_height = primary_height * weight_primary + secondary_height * weight_secondary
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

# 498行目の後に挿入（0-indexed: 497の後）
result_lines = lines[:497] + [new_interpolation_code] + lines[497:]

with open('terrain_voxel_v11_smooth_complete.py', 'w', encoding='utf-8') as f:
    f.writelines(result_lines)

print("✅ 区分境界完全滑らか版作成: terrain_voxel_v11_smooth_complete.py")
print("\n改善内容:")
print("  - 角度比による完全な重み付け平均を実装")
print("  - weight_primary = angle_to_secondary / total_angle")
print("  - weight_secondary = angle_to_primary / total_angle")
print("  - 区分境界で連続的に遷移（段差なし）")
