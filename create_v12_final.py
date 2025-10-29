#!/usr/bin/env python3
"""v12最終版を作成：マップ拡大+完全滑らか補間+比率調整"""

with open('terrain_voxel_v10_adjusted.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 新しい補間コード（完全な角度比重み付け）
new_interpolation = '''                # === ステップ3: 区分境界での完全な滑らかな補間 ===
                if len(boundary_info) > 1:
                    secondary_boundary = boundary_info[1]
                    primary_key = primary_boundary['key']
                    primary_bc = self.boundary_centroids[primary_key]
                    primary_vec = np.array([primary_bc[0] - centroid[0], primary_bc[1] - centroid[1]])
                    primary_vec_norm = np.linalg.norm(primary_vec)

                    secondary_key = secondary_boundary['key']
                    secondary_bc = self.boundary_centroids[secondary_key]
                    secondary_vec = np.array([secondary_bc[0] - centroid[0], secondary_bc[1] - centroid[1]])
                    secondary_vec_norm = np.linalg.norm(secondary_vec)

                    if primary_vec_norm > 0.1 and secondary_vec_norm > 0.1:
                        cross_ps = primary_vec[0] * secondary_vec[1] - primary_vec[1] * secondary_vec[0]
                        cross_pp = primary_vec[0] * pixel_vec[1] - primary_vec[1] * pixel_vec[0]
                        cross_xs = pixel_vec[0] * secondary_vec[1] - pixel_vec[1] * secondary_vec[0]

                        in_between = (np.sign(cross_ps) == np.sign(cross_pp) and np.sign(cross_ps) == np.sign(cross_xs))
                        almost_zero = abs(cross_ps) < 0.1 or abs(cross_pp) < 0.1 or abs(cross_xs) < 0.1

                        if in_between or almost_zero:
                            cos_pixel_primary = np.dot(pixel_vec, primary_vec) / (pixel_norm * primary_vec_norm)
                            angle_to_primary = np.arccos(np.clip(cos_pixel_primary, -1.0, 1.0))

                            cos_pixel_secondary = np.dot(pixel_vec, secondary_vec) / (pixel_norm * secondary_vec_norm)
                            angle_to_secondary = np.arccos(np.clip(cos_pixel_secondary, -1.0, 1.0))

                            total_angle = angle_to_primary + angle_to_secondary
                            if total_angle > 0.01:
                                weight_primary = angle_to_secondary / total_angle
                                weight_secondary = angle_to_primary / total_angle

                                secondary_neighbor_height = self.group_heights[secondary_boundary['neighbor_id']]
                                secondary_distance = secondary_boundary['distance']
                                secondary_is_wall = self.is_wall.get(secondary_key, True)

                                if secondary_is_wall:
                                    if secondary_distance < 15.0:
                                        t = secondary_distance / 15.0
                                        sec_blend = 1.0 - (t ** 0.3)
                                    else:
                                        sec_blend = 0.0
                                    if base_height > secondary_neighbor_height:
                                        secondary_height = base_height * (1.0 - sec_blend) + secondary_neighbor_height * sec_blend
                                    else:
                                        secondary_height = base_height
                                else:
                                    if secondary_distance < 60.0:
                                        t = secondary_distance / 60.0
                                        sec_blend = 1.0 - (t ** 0.8)
                                    else:
                                        sec_blend = 0.0
                                    sec_boundary_height = (base_height + secondary_neighbor_height) / 2.0
                                    secondary_height = base_height * (1.0 - sec_blend) + sec_boundary_height * sec_blend

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

# 500-580行目を置き換え（0-indexed: 499-579）
result_lines = lines[:499] + [new_interpolation] + lines[580:]

# 文字列として結合
content = ''.join(result_lines)

# マップサイズと比率の調整
content = content.replace('width=100', 'width=200')
content = content.replace('height=100', 'height=200')
content = content.replace('self.width = 100', 'self.width = 200')
content = content.replace('self.height = 100', 'self.height = 200')
content = content.replace('num_cells=200', 'num_cells=350')
content = content.replace('num_groups=15', 'num_groups=25')
content = content.replace('num_additional_passages = int(len(all_walls) * 0.4)',
                         'num_additional_passages = int(len(all_walls) * 0.25)')
content = content.replace('min_diff = 12', 'min_diff = 15')
content = content.replace('cliff_range = 12.0', 'cliff_range = 15.0')
content = content.replace('slope_range = 45.0', 'slope_range = 60.0')

with open('terrain_voxel_v12_final.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ v12最終版作成: terrain_voxel_v12_final.py")
print("\n全変更内容:")
print("  【滑らか化】角度比による完全な重み付け平均")
print("  【マップ】100x100 → 200x200（4倍）")
print("  【領域】セル350、グループ25")
print("  【崖増加】追加通路40% → 25%")
print("  【明確化】壁最低高度差12m → 15m")
print("  【範囲】cliff 15m、slope 60m")
