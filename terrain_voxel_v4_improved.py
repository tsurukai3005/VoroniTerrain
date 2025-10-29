"""
等距離線ベースの高度マップ生成（改善版v4）

修正ポイント:
1. 等距離線に沿った高度変化を正しく実装
2. 坂: 中心から境界まで線形的に高度が変化
3. 崖: 境界付近(30%範囲)で急激に変化、中央部はほぼ平坦
4. 方向による滑らかな補間で不連続を解消
"""

def _create_distance_based_height_map(self):
    """等距離線ベースの高度マップ生成（v4改善版）"""
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

            # 境界からの距離情報を取得
            dist_to_boundary_map = self.distance_to_boundary.get(current_group, {})
            if not dist_to_boundary_map:
                height_map[y, x] = base_height
                continue

            # このピクセルから全境界への最短距離
            min_dist_to_any_boundary = dist_to_boundary_map.get((x, y), None)
            if min_dist_to_any_boundary is None:
                height_map[y, x] = base_height
                continue

            # 領域の最大距離（中心付近）を取得
            max_distance = max(dist_to_boundary_map.values()) if dist_to_boundary_map else 1.0

            # 正規化距離（0=境界、1=中心）
            if max_distance > 0:
                normalized_dist = min_dist_to_any_boundary / max_distance
            else:
                normalized_dist = 0.0

            # 各境界方向からの寄与を計算
            dist_field = self.distance_fields.get(current_group, {})
            if not dist_field:
                height_map[y, x] = base_height
                continue

            contributions = []
            total_weight = 0.0

            for neighbor_id, pixel_distances in dist_field.items():
                dist_to_this_boundary = pixel_distances.get((x, y), float('inf'))

                if dist_to_this_boundary == float('inf'):
                    continue

                key = tuple(sorted([current_group, neighbor_id]))
                if key not in self.boundary_centroids:
                    continue

                # 境界の種類を判定
                is_wall = self.is_wall.get(key, True)
                neighbor_height = self.group_heights[neighbor_id]

                # 境界中央の位置
                boundary_center = self.boundary_centroids[key]
                boundary_vec = np.array([boundary_center[0] - centroid[0],
                                        boundary_center[1] - centroid[1]])
                boundary_norm = np.linalg.norm(boundary_vec)

                if pixel_norm < 0.1 or boundary_norm < 0.1:
                    continue

                # 方向の一致度（cos類似度）
                cos_angle = np.dot(pixel_vec, boundary_vec) / (pixel_norm * boundary_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)

                # 方向が合わない場合はスキップ（72度以上離れている）
                if cos_angle < 0.3:
                    continue

                # === 等距離線に沿った高度変化を計算 ===
                # この境界方向での正規化距離
                normalized_dist_to_boundary = dist_to_this_boundary / max_distance if max_distance > 0 else 0.0

                if is_wall:
                    # 崖: 境界付近（0-0.3の範囲）で急激に変化
                    if normalized_dist_to_boundary < 0.3:
                        # 0.0 → 0.3 を 1.0 → 0.0 に変換（べき乗で急峻に）
                        t = normalized_dist_to_boundary / 0.3
                        distance_weight = 1.0 - (t ** 0.3)  # 急峻な減衰
                    else:
                        # 中央部: 影響なし
                        distance_weight = 0.0
                else:
                    # 坂: 全範囲（0-1.0）で線形的に変化
                    distance_weight = 1.0 - normalized_dist_to_boundary  # 線形減衰

                # 方向による重み（境界方向に近いほど大きい）
                # cos_angle: 0.3-1.0 を 0.0-1.0 に正規化
                angle_weight = (cos_angle - 0.3) / 0.7
                angle_weight = max(0.0, min(1.0, angle_weight))
                angle_weight = angle_weight ** 1.5  # 方向性を強調

                # 最終的な寄与度
                contribution = distance_weight * angle_weight

                if contribution > 0.001:  # 微小な寄与は無視
                    contributions.append({
                        'weight': contribution,
                        'neighbor_height': neighbor_height
                    })
                    total_weight += contribution

            # === すべての境界からの寄与を統合 ===
            if contributions and total_weight > 0:
                # 重み付き平均で最終高度を計算
                weighted_height = 0.0
                for contrib in contributions:
                    normalized_weight = contrib['weight'] / total_weight
                    # base_height から neighbor_height への補間
                    blend_height = base_height + (contrib['neighbor_height'] - base_height) * contrib['weight']
                    weighted_height += blend_height * normalized_weight

                height_map[y, x] = weighted_height
            else:
                # 影響範囲外（中央部）: ベース高度
                height_map[y, x] = base_height

    return height_map
