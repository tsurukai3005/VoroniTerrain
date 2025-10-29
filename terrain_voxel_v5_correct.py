"""
正しい等距離線ベースの高度マップ生成

アルゴリズム:
1. 各ピクセルは重心からの方向で最も近い境界に属する（区分）
2. その境界までの実距離（ピクセル単位）で高度を計算
3. 区分境界付近では隣接区分と角度で滑らかに補間
"""

def _create_distance_based_height_map_correct(self):
    """等距離線ベースの高度マップ生成（v5修正版）"""
    height_map = np.zeros((self.height, self.width))

    # 重心を計算
    group_centroids = {}
    for group_id in range(self.num_groups):
        y_coords, x_coords = np.where(self.group_map == group_id)
        if len(x_coords) > 0:
            group_centroids[group_id] = (np.mean(x_coords), np.mean(y_coords))

    for y in range(self.height):
        for x in range(self.width):
            current_group = self.group_map[y, x]
            base_height = self.group_heights[current_group]

            if current_group not in group_centroids:
                height_map[y, x] = base_height
                continue

            centroid = group_centroids[current_group]
            pixel_vec = np.array([x - centroid[0], y - centroid[1]])
            pixel_norm = np.linalg.norm(pixel_vec)

            if pixel_norm < 0.1:
                # 重心付近: ベース高度
                height_map[y, x] = base_height
                continue

            # 各境界への情報を取得
            dist_field = self.distance_fields.get(current_group, {})
            if not dist_field:
                height_map[y, x] = base_height
                continue

            # === ステップ1: 最も近い境界方向を特定（主境界） ===
            boundary_info = []

            for neighbor_id, pixel_distances in dist_field.items():
                dist_to_this_boundary = pixel_distances.get((x, y), float('inf'))

                if dist_to_this_boundary == float('inf'):
                    continue

                key = tuple(sorted([current_group, neighbor_id]))
                if key not in self.boundary_centroids:
                    continue

                # 境界中央への方向ベクトル
                boundary_center = self.boundary_centroids[key]
                boundary_vec = np.array([boundary_center[0] - centroid[0],
                                        boundary_center[1] - centroid[1]])
                boundary_norm = np.linalg.norm(boundary_vec)

                if boundary_norm < 0.1:
                    continue

                # ピクセルと境界方向の角度一致度
                cos_angle = np.dot(pixel_vec, boundary_vec) / (pixel_norm * boundary_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)

                # 情報を保存
                boundary_info.append({
                    'neighbor_id': neighbor_id,
                    'distance': dist_to_this_boundary,
                    'cos_angle': cos_angle,
                    'key': key
                })

            if not boundary_info:
                height_map[y, x] = base_height
                continue

            # cos_angleが最大（方向が最も一致）の境界を主境界とする
            boundary_info.sort(key=lambda b: b['cos_angle'], reverse=True)
            primary_boundary = boundary_info[0]

            # 主境界のcos_angleが負（90度以上離れている）場合は影響なし
            if primary_boundary['cos_angle'] < 0.0:
                height_map[y, x] = base_height
                continue

            # === ステップ2: 主境界までの距離で高度を計算 ===
            primary_key = primary_boundary['key']
            is_wall = self.is_wall.get(primary_key, True)
            neighbor_height = self.group_heights[primary_boundary['neighbor_id']]
            distance = primary_boundary['distance']

            if is_wall:
                # 崖: 境界から8ピクセル以内で急激に変化
                cliff_range = 8.0
                if distance < cliff_range:
                    t = distance / cliff_range
                    # べき乗0.2 = 急峻（境界でストンと落ちる）
                    blend = 1.0 - (t ** 0.2)
                else:
                    # 崖の範囲外: ベース高度を保つ
                    blend = 0.0
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

            # 主境界からの高度
            primary_height = base_height * (1.0 - blend) + neighbor_height * blend

            # === ステップ3: 隣接境界との補間（区分境界を滑らかに） ===
            if len(boundary_info) > 1:
                # 2番目に近い方向の境界
                secondary_boundary = boundary_info[1]

                # 2つの境界方向の間の角度差
                angle_diff = np.arccos(np.clip(primary_boundary['cos_angle'], -1.0, 1.0)) - \
                            np.arccos(np.clip(secondary_boundary['cos_angle'], -1.0, 1.0))

                # 角度差が小さい（30度以内）場合は補間
                if abs(angle_diff) < np.pi / 6:  # 30度
                    secondary_key = secondary_boundary['key']
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

                    # 角度差に応じて補間（角度差が小さいほど補間が強い）
                    interpolation_weight = 1.0 - abs(angle_diff) / (np.pi / 6)
                    interpolation_weight = interpolation_weight ** 2  # 境界付近で強く補間

                    # 主境界と副境界の高度を補間
                    final_height = primary_height * (1.0 - interpolation_weight * 0.3) + \
                                  secondary_height * (interpolation_weight * 0.3)

                    height_map[y, x] = final_height
                else:
                    # 角度差が大きい: 主境界のみ
                    height_map[y, x] = primary_height
            else:
                # 境界が1つのみ: 主境界のみ
                height_map[y, x] = primary_height

    return height_map
