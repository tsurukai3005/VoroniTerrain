# 修正版の高度計算メソッド（490-521行目の置き換え）

# === ステップ3: 隣接区分との補間（区分境界を滑らかに） ===
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

        # pixel方向が2つの境界の「間」にあるかチェック（外積で判定）
        # 2D外積: a×b = ax*by - ay*bx
        cross_primary_secondary = primary_vec[0] * secondary_vec[1] - primary_vec[1] * secondary_vec[0]
        cross_primary_pixel = primary_vec[0] * pixel_vec[1] - primary_vec[1] * pixel_vec[0]
        cross_pixel_secondary = pixel_vec[0] * secondary_vec[1] - pixel_vec[1] * secondary_vec[0]

        # pixelが2つの境界の間にある条件：
        # cross_primary_secondary と cross_primary_pixel が同符号
        # かつ cross_primary_secondary と cross_pixel_secondary が同符号
        sign_check = (np.sign(cross_primary_secondary) == np.sign(cross_primary_pixel) and
                     np.sign(cross_primary_secondary) == np.sign(cross_pixel_secondary))

        # または、両方が非常に小さい（ほぼ一直線）
        almost_zero = abs(cross_primary_secondary) < 0.1 or abs(cross_primary_pixel) < 0.1 or abs(cross_pixel_secondary) < 0.1

        if sign_check or almost_zero:
            # pixel方向からprimary方向への角度
            cos_pixel_primary = np.dot(pixel_vec, primary_vec) / (pixel_norm * primary_vec_norm)
            cos_pixel_primary = np.clip(cos_pixel_primary, -1.0, 1.0)
            angle_pixel_primary = np.arccos(cos_pixel_primary)

            # 補間重み（0.0 = 完全にprimary, 1.0 = 完全にsecondary）
            if boundaries_angle > 0.01:  # ゼロ除算回避
                interpolation_weight = angle_pixel_primary / boundaries_angle
                interpolation_weight = np.clip(interpolation_weight, 0.0, 1.0)

                # secondary境界の高度を計算
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
                else:
                    slope_range = 30.0
                    if secondary_distance < slope_range:
                        t = secondary_distance / slope_range
                        secondary_blend = 1.0 - (t ** 0.7)
                    else:
                        secondary_blend = 0.0

                secondary_height = base_height * (1.0 - secondary_blend) + secondary_neighbor_height * secondary_blend

                # 区分境界での滑らかな補間（重みを緩和）
                smooth_weight = interpolation_weight * 0.5  # 補間の影響を50%に制限
                final_height = primary_height * (1.0 - smooth_weight) + secondary_height * smooth_weight
            else:
                final_height = primary_height
        else:
            # 区分の間にない場合、primary高度のみ使用
            final_height = primary_height
    else:
        final_height = primary_height
else:
    # 境界が1つしかない場合
    final_height = primary_height

height_map[y, x] = final_height
