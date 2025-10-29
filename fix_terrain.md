境界の中央を中心とした同心円が等距離線として描かれていますが、これは想定しているものではないです。
等距離線は領域の外周の形状を内側にマトリョーシカのように収めていった形状になります。なので等距離線同士が交わることはありません。等距離線を生成する作業と境界の中央点は関係ないです。等距離線上の高さを計算するときに境界中心方向をまずは考え、各坂の方向の間を補正していくイメージです。
理解しました！等距離線の生成方法を誤解していました。 正しい理解:
等距離線 = 境界からの距離が等しい点の集合（境界形状に沿ったマトリョーシカ）
境界中心は等距離線の生成には無関係
境界中心方向は高度の補間計算にのみ使用
現在の問題: 各境界ごとに別々の距離フィールドを作っているため、同心円のような形になっている 修正方針:
グループ全体の境界からの距離フィールドを計算（全境界の最短距離）
その距離フィールドで等距離線を描画 → マトリョーシカ形状
高度計算では、各境界方向ごとに異なる高度を設定し、境界中心方向で補間
修正します:
Edit Notebook Cell f:\Github\agi-agents\.tmp\terrain_voxel_v3.ipynb:cell-3
⎿
Success
class VoronoiTerrain:
    def __init__(self, width=100, height=100, num_sites=200, num_groups=15):
        self.width = width
        self.height = height
        self.num_sites = num_sites
        self.num_groups = num_groups
        
        self.points = self._generate_relaxed_points()
        self.vor = Voronoi(self.points)
        self.cell_map = self._build_cell_map()
        self.group_map = self._create_groups()
        self.group_edges = self._build_group_edges()
        self._generate_maze()
        self.group_heights = self._propagate_heights()
        
        # 境界ピクセルと距離フィールドを計算
        self._compute_boundary_pixels_and_distance_field()
        
        # 境界の中央点を計算
        self._compute_boundary_centroids()
        
        # 角度ベースの高度マップ
        self.height_map = self._create_angle_based_height_map()
    
    def _generate_relaxed_points(self, iterations=3):
        """Lloyd緩和でより均等な点配置"""
        points = np.random.rand(self.num_sites, 2)
        points[:, 0] *= self.width
        points[:, 1] *= self.height
        
        for _ in range(iterations):
            vor = Voronoi(points)
            centroids = []
            
            for i, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                if -1 in region or len(region) == 0:
                    centroids.append(points[i])
                    continue
                
                vertices = vor.vertices[region]
                centroid = vertices.mean(axis=0)
                centroid[0] = np.clip(centroid[0], 0, self.width)
                centroid[1] = np.clip(centroid[1], 0, self.height)
                centroids.append(centroid)
            
            points = np.array(centroids)
        
        return points
    
    def _build_cell_map(self):
        """各ピクセルに最も近いセルIDを割り当て"""
        cell_map = np.zeros((self.height, self.width), dtype=int)
        
        for y in range(self.height):
            for x in range(self.width):
                distances = np.sqrt((self.points[:, 0] - x)**2 + (self.points[:, 1] - y)**2)
                cell_map[y, x] = np.argmin(distances)
        
        return cell_map
    
    def _create_groups(self):
        """セルをグループ化（母点数で小:中:大 = 1:3:9、出現確率均等）"""
        num_each = self.num_groups // 3
        remainder = self.num_groups % 3
        
        size_types = []
        size_types.extend(['small'] * (num_each + (1 if remainder > 0 else 0)))
        size_types.extend(['medium'] * (num_each + (1 if remainder > 1 else 0)))
        size_types.extend(['large'] * num_each)
        
        random.shuffle(size_types)
        
        cells_per_group = self.num_sites / self.num_groups
        target_sizes = []
        for size_type in size_types:
            if size_type == 'small':
                target_sizes.append(int(cells_per_group * 1.0))
            elif size_type == 'medium':
                target_sizes.append(int(cells_per_group * 3.0))
            else:
                target_sizes.append(int(cells_per_group * 9.0))
        
        seeds = [random.randint(0, self.num_sites - 1)]
        
        while len(seeds) < self.num_groups:
            max_min_dist = -1
            best_candidate = -1
            
            for i in range(self.num_sites):
                if i in seeds:
                    continue
                
                min_dist = min([np.linalg.norm(self.points[i] - self.points[s]) for s in seeds])
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = i
            
            if best_candidate != -1:
                seeds.append(best_candidate)
            else:
                break
        
        cell_to_group = {}
        group_cells = {i: set() for i in range(self.num_groups)}
        
        for i, seed in enumerate(seeds):
            cell_to_group[seed] = i
            group_cells[i].add(seed)
        
        neighbors = {i: set() for i in range(self.num_sites)}
        for ridge in self.vor.ridge_points:
            neighbors[ridge[0]].add(ridge[1])
            neighbors[ridge[1]].add(ridge[0])
        
        queue = deque([(seed, i) for i, seed in enumerate(seeds)])
        visited = set(seeds)
        
        while queue:
            cell_id, group_id = queue.popleft()
            
            if len(group_cells[group_id]) >= target_sizes[group_id]:
                continue
            
            for neighbor in neighbors[cell_id]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    cell_to_group[neighbor] = group_id
                    group_cells[group_id].add(neighbor)
                    queue.append((neighbor, group_id))
        
        for i in range(self.num_sites):
            if i not in cell_to_group:
                min_dist = float('inf')
                closest_group = 0
                
                for group_id, cells in group_cells.items():
                    for cell in cells:
                        dist = np.linalg.norm(self.points[i] - self.points[cell])
                        if dist < min_dist:
                            min_dist = dist
                            closest_group = group_id
                
                cell_to_group[i] = closest_group
                group_cells[closest_group].add(i)
        
        group_map = np.zeros((self.height, self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                cell_id = self.cell_map[y, x]
                group_map[y, x] = cell_to_group[cell_id]
        
        self.cell_to_group = cell_to_group
        self.group_cells = group_cells
        
        return group_map
    
    def _build_group_edges(self):
        """グループ間の隣接関係を構築"""
        edges = {i: set() for i in range(self.num_groups)}
        
        for ridge in self.vor.ridge_points:
            g1 = self.cell_to_group[ridge[0]]
            g2 = self.cell_to_group[ridge[1]]
            
            if g1 != g2:
                edges[g1].add(g2)
                edges[g2].add(g1)
        
        return edges
    
    def _generate_maze(self):
        """グループ間で迷路を生成（通路と壁）- より開けた構造"""
        visited = {0}
        edges = [(0, neighbor) for neighbor in self.group_edges[0]]
        self.is_wall = {}
        
        while len(visited) < self.num_groups and edges:
            idx = random.randint(0, len(edges) - 1)
            from_group, to_group = edges.pop(idx)
            
            if to_group not in visited:
                visited.add(to_group)
                key = tuple(sorted([from_group, to_group]))
                self.is_wall[key] = False
                
                for neighbor in self.group_edges[to_group]:
                    if neighbor not in visited:
                        edges.append((to_group, neighbor))
        
        for g1 in range(self.num_groups):
            for g2 in self.group_edges[g1]:
                key = tuple(sorted([g1, g2]))
                if key not in self.is_wall:
                    self.is_wall[key] = True
        
        all_walls = [key for key, is_wall in self.is_wall.items() if is_wall]
        num_additional_passages = int(len(all_walls) * 0.4)
        
        for _ in range(num_additional_passages):
            if all_walls:
                wall_to_remove = random.choice(all_walls)
                self.is_wall[wall_to_remove] = False
                all_walls.remove(wall_to_remove)
    
    def _propagate_heights(self):
        """グループごとに高度を伝播（通路は2-4mの小さな高度差）"""
        heights = {}
        heights[0] = 40 + random.random() * 20
        
        visited = {0}
        queue = deque([(0, None)])
        
        while queue:
            current, last_direction = queue.popleft()
            
            for neighbor in self.group_edges[current]:
                key = tuple(sorted([current, neighbor]))
                
                if not self.is_wall[key] and neighbor not in visited:
                    visited.add(neighbor)
                    
                    height_diff = 2 + random.random() * 2
                    
                    if last_direction == 'up':
                        up_prob = 0.75
                    elif last_direction == 'down':
                        up_prob = 0.25
                    else:
                        up_prob = 0.55
                    
                    go_up = random.random() < up_prob
                    new_direction = 'up' if go_up else 'down'
                    
                    if go_up:
                        heights[neighbor] = min(90, heights[current] + height_diff)
                    else:
                        heights[neighbor] = max(10, heights[current] - height_diff)
                    
                    queue.append((neighbor, new_direction))
        
        avg_height = np.mean(list(heights.values()))
        for i in range(self.num_groups):
            if i not in heights:
                heights[i] = avg_height
        
        self._ensure_wall_height_difference(heights)
        
        return heights
    
    def _ensure_wall_height_difference(self, heights):
        """壁の両側で最低15mの高度差を確保"""
        min_diff = 15
        
        for key, is_wall in self.is_wall.items():
            if is_wall:
                g1, g2 = key
                diff = abs(heights[g1] - heights[g2])
                
                if diff < min_diff:
                    if heights[g1] > heights[g2]:
                        heights[g1] = min(95, heights[g2] + min_diff + random.random() * 10)
                    else:
                        heights[g2] = min(95, heights[g1] + min_diff + random.random() * 10)
    
    def _compute_boundary_pixels_and_distance_field(self):
        """境界ピクセルの検出と距離フィールドの計算"""
        self.group_boundaries = {i: {} for i in range(self.num_groups)}
        
        # 境界ピクセルを検出
        for y in range(self.height):
            for x in range(self.width):
                current_group = self.group_map[y, x]
                
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        neighbor_group = self.group_map[ny, nx]
                        
                        if neighbor_group != current_group:
                            if neighbor_group not in self.group_boundaries[current_group]:
                                self.group_boundaries[current_group][neighbor_group] = []
                            
                            self.group_boundaries[current_group][neighbor_group].append((x, y))
        
        # 距離フィールド: グループごとに全境界への最短距離
        self.distance_fields = {}
        self.distance_to_boundary = {}  # マトリョーシカ用：グループ全境界への最短距離
        
        for group_id in range(self.num_groups):
            # 各境界への個別距離（高度計算用）
            dist_field = {}
            
            # グループの全境界ピクセルを収集
            all_boundary_pixels = []
            for neighbor_id, boundary_pixels in self.group_boundaries[group_id].items():
                all_boundary_pixels.extend(boundary_pixels)
                
                if len(boundary_pixels) == 0:
                    continue
                
                boundary_array = np.array(boundary_pixels)
                y_coords, x_coords = np.where(self.group_map == group_id)
                distances = np.full(len(x_coords), float('inf'))
                
                for i, (px, py) in enumerate(zip(x_coords, y_coords)):
                    dists = np.sqrt((boundary_array[:, 0] - px)**2 + (boundary_array[:, 1] - py)**2)
                    distances[i] = np.min(dists)
                
                dist_field[neighbor_id] = {}
                for i, (px, py) in enumerate(zip(x_coords, y_coords)):
                    dist_field[neighbor_id][(px, py)] = distances[i]
            
            self.distance_fields[group_id] = dist_field
            
            # グループ全境界への最短距離（マトリョーシカ用）
            if len(all_boundary_pixels) > 0:
                all_boundary_array = np.array(all_boundary_pixels)
                y_coords, x_coords = np.where(self.group_map == group_id)
                
                overall_distances = np.full(len(x_coords), float('inf'))
                for i, (px, py) in enumerate(zip(x_coords, y_coords)):
                    dists = np.sqrt((all_boundary_array[:, 0] - px)**2 + 
                                   (all_boundary_array[:, 1] - py)**2)
                    overall_distances[i] = np.min(dists)
                
                # 距離マップを保存
                dist_map = {}
                for i, (px, py) in enumerate(zip(x_coords, y_coords)):
                    dist_map[(px, py)] = overall_distances[i]
                
                self.distance_to_boundary[group_id] = dist_map
    
    def _compute_boundary_centroids(self):
        """各境界の中央点（境界線上の中点）を計算"""
        self.boundary_centroids = {}
        
        for group_id in range(self.num_groups):
            for neighbor_id, boundary_pixels in self.group_boundaries[group_id].items():
                if len(boundary_pixels) == 0:
                    continue
                
                boundary_array = np.array(boundary_pixels)
                centroid_x = np.mean(boundary_array[:, 0])
                centroid_y = np.mean(boundary_array[:, 1])
                
                key = tuple(sorted([group_id, neighbor_id]))
                self.boundary_centroids[key] = (centroid_x, centroid_y)
    
    def _create_angle_based_height_map(self):
        """角度ベースの高度マップ生成"""
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
                
                dist_field = self.distance_fields.get(current_group, {})
                
                if not dist_field:
                    height_map[y, x] = base_height
                    continue
                
                # 各境界方向の高度を計算
                angle_heights = []
                
                for neighbor_id, pixel_distances in dist_field.items():
                    dist = pixel_distances.get((x, y), float('inf'))
                    
                    if dist == float('inf'):
                        continue
                    
                    key = tuple(sorted([current_group, neighbor_id]))
                    if key not in self.boundary_centroids:
                        continue
                    
                    boundary_center = self.boundary_centroids[key]
                    boundary_vec = np.array([boundary_center[0] - centroid[0], 
                                            boundary_center[1] - centroid[1]])
                    
                    pixel_norm = np.linalg.norm(pixel_vec)
                    boundary_norm = np.linalg.norm(boundary_vec)
                    
                    if pixel_norm < 0.1 or boundary_norm < 0.1:
                        continue
                    
                    cos_angle = np.dot(pixel_vec, boundary_vec) / (pixel_norm * boundary_norm)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    
                    is_wall = self.is_wall.get(key, True)
                    neighbor_height = self.group_heights[neighbor_id]
                    height_diff = abs(base_height - neighbor_height)
                    
                    if is_wall:
                        influence_range = 8.0 + height_diff * 0.5
                        power = 2.5
                    else:
                        influence_range = 20.0 + height_diff * 4.0
                        power = 0.4 + height_diff * 0.015
                    
                    if dist < influence_range:
                        t = dist / influence_range
                        dist_weight = 1.0 - (t ** power)
                        
                        # 角度による重み（境界中心方向で最大）
                        angle_weight = (cos_angle + 1.0) / 2.0
                        angle_weight = angle_weight ** 2
                        
                        combined_weight = dist_weight * angle_weight
                        
                        angle_heights.append({
                            'weight': combined_weight,
                            'height': neighbor_height
                        })
                
                # 重み付き平均
                if angle_heights:
                    total_weight = 1.0
                    weighted_height = base_height * 1.0
                    
                    for ah in angle_heights:
                        total_weight += ah['weight']
                        weighted_height += ah['height'] * ah['weight']
                    
                    height_map[y, x] = weighted_height / total_weight
                else:
                    height_map[y, x] = base_height
        
        return height_map

print("VoronoiTerrainクラスの定義完了")
次に、可視化を修正してdistance_to_boundaryを使用してマトリョーシカ形状の等距離線を描画します:
Edit Notebook Cell f:\Github\agi-agents\.tmp\terrain_voxel_v3.ipynb:cell-7
⎿
Success
# カスタムカラーマップ:青→オレンジ
colors_blue_orange = ['#0066cc', '#3399ff', '#99ccff', '#ffcc99', '#ff9933', '#ff6600']
n_bins = 100
cmap_blue_orange = LinearSegmentedColormap.from_list('blue_orange', colors_blue_orange, N=n_bins)

# グループ重心を計算
group_centroids = {}
for group_id in range(terrain.num_groups):
    y_coords, x_coords = np.where(terrain.group_map == group_id)
    if len(x_coords) > 0:
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        group_centroids[group_id] = (centroid_x, centroid_y)

# グループ構造の可視化（改善版）
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# === 左: グループマップ（高さと境界種別を表示）===
ax1 = axes[0]

# 各グループの高さで色分け（透明度を上げて薄く）
height_based_map = np.zeros((terrain.height, terrain.width))
for y in range(terrain.height):
    for x in range(terrain.width):
        group_id = terrain.group_map[y, x]
        height_based_map[y, x] = terrain.group_heights[group_id]

im1 = ax1.imshow(height_based_map, cmap=cmap_blue_orange, interpolation='nearest', alpha=0.3)

# 境界ピクセルを収集
boundary_segments = {'wall': [], 'passage': []}

for y in range(terrain.height):
    for x in range(terrain.width):
        current_group = terrain.group_map[y, x]
        
        # 右と下の隣接ピクセルをチェック
        for dy, dx in [(0, 1), (1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < terrain.height and 0 <= nx < terrain.width:
                neighbor_group = terrain.group_map[ny, nx]
                if neighbor_group != current_group:
                    key = tuple(sorted([current_group, neighbor_group]))
                    is_wall = terrain.is_wall.get(key, False)
                    segment_type = 'wall' if is_wall else 'passage'
                    boundary_segments[segment_type].append(([x, nx], [y, ny]))

# 境界を太い実線で描画
for segment_type, color in [('wall', 'red'), ('passage', 'green')]:
    for xs, ys in boundary_segments[segment_type]:
        ax1.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, solid_capstyle='round')

# 重心に点を描画
for group_id, centroid in group_centroids.items():
    ax1.plot(centroid[0], centroid[1], 'ko', markersize=4)

# 通路方向への矢印（緑）と壁方向への矢印（赤）
for key, is_wall in terrain.is_wall.items():
    g1, g2 = key
    if g1 in group_centroids and g2 in group_centroids:
        if key not in terrain.boundary_centroids:
            continue
        
        c1 = group_centroids[g1]
        c2 = group_centroids[g2]
        boundary_center = terrain.boundary_centroids[key]
        
        color = 'red' if is_wall else 'green'
        arrow_alpha = 0.7
        
        # g1から境界中央への矢印
        dx1 = boundary_center[0] - c1[0]
        dy1 = boundary_center[1] - c1[1]
        ax1.arrow(c1[0], c1[1], dx1*0.8, dy1*0.8, 
                 head_width=2, head_length=2, fc=color, ec=color, alpha=arrow_alpha, linewidth=1.5)
        
        # g2から境界中央への矢印
        dx2 = boundary_center[0] - c2[0]
        dy2 = boundary_center[1] - c2[1]
        ax1.arrow(c2[0], c2[1], dx2*0.8, dy2*0.8, 
                 head_width=2, head_length=2, fc=color, ec=color, alpha=arrow_alpha, linewidth=1.5)

ax1.set_title('グループ分割（坂の方向を矢印で表示）\n緑=通路(緩やかな坂)、赤=壁(崖)', fontsize=14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
plt.colorbar(im1, ax=ax1, label='高度 (m)')

# === 右: 高度マップ（境界と等距離線を描画）===
ax2 = axes[1]
im2 = ax2.imshow(terrain.height_map, cmap=cmap_blue_orange, interpolation='nearest')

# 境界を太い黒実線で描画
for xs, ys in boundary_segments['wall'] + boundary_segments['passage']:
    ax2.plot(xs, ys, color='black', linewidth=2.0, alpha=1.0, solid_capstyle='round')

# 等距離線を細線で描画（マトリョーシカ形状）
for group_id in range(terrain.num_groups):
    # グループ全境界への距離マップ（マトリョーシカ用）
    dist_to_boundary = terrain.distance_to_boundary.get(group_id, {})
    
    if len(dist_to_boundary) == 0:
        continue
    
    # 距離マップを2D配列に変換
    dist_map = np.full((terrain.height, terrain.width), np.nan)
    for (px, py), dist in dist_to_boundary.items():
        dist_map[py, px] = dist
    
    # matplotlibのcontourを使用して等距離線を描画
    x_grid = np.arange(terrain.width)
    y_grid = np.arange(terrain.height)
    
    # 等距離線を描画（5, 10, 15, 20, 25ピクセル）
    for target_dist in [5, 10, 15, 20, 25]:
        valid_mask = ~np.isnan(dist_map)
        
        if valid_mask.sum() > 10:
            try:
                contour_set = ax2.contour(x_grid, y_grid, dist_map, 
                                         levels=[target_dist], 
                                         colors='black', 
                                         linewidths=0.5, 
                                         alpha=0.4)
            except:
                pass

# 重心と境界中央を結ぶ線を描画（折れ線）
for group_id in range(terrain.num_groups):
    if group_id not in group_centroids:
        continue
    
    centroid = group_centroids[group_id]
    
    # この群の各境界について
    for neighbor_id in terrain.group_boundaries[group_id].keys():
        key = tuple(sorted([group_id, neighbor_id]))
        if key not in terrain.boundary_centroids:
            continue
        
        boundary_center = terrain.boundary_centroids[key]
        
        # 重心から境界中央への線（青い破線）
        ax2.plot([centroid[0], boundary_center[0]], [centroid[1], boundary_center[1]], 
                'b--', linewidth=1, alpha=0.6)
        
        # 境界中央から隣接グループ重心への線（青い破線）
        if neighbor_id in group_centroids:
            neighbor_centroid = group_centroids[neighbor_id]
            ax2.plot([boundary_center[0], neighbor_centroid[0]], 
                    [boundary_center[1], neighbor_centroid[1]], 
                    'b--', linewidth=1, alpha=0.6)

ax2.set_title('高度マップ（境界=黒太線、等距離線=黒細線、重心↔境界中央=青破線）', fontsize=14)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(im2, ax=ax2, label='高度 (m)')

plt.tight_layout()
plt.show()
修正が完了しました。主な変更点:
cell-3の変更
新しい距離フィールド構造:
self.distance_fields: 各境界への個別距離（高度計算用）- 既存の実装
self.distance_to_boundary: グループ全境界への最短距離（マトリョーシカ形状の等距離線用）- 新規追加
_compute_boundary_pixels_and_distance_field()の改善:
グループの全境界ピクセルを1つの配列にまとめる
各ピクセルから全境界への最短距離を計算
distance_to_boundary[group_id][(x,y)] = 最短距離として保存
cell-7の変更
等距離線の描画:
terrain.distance_to_boundaryを使用（各グループごとに1つの距離マップ）
この距離マップは境界形状に沿ったマトリョーシカ形状になります
各グループで等距離線が境界の形状に沿って描画されます
期待される結果
等距離線がマトリョーシカ形状: グループの境界形状に沿って内側に収まっていく輪
等距離線同士が交わらない: 各グループ内で境界からの距離が増えるごとに内側に縮小
境界中心は高度計算のみに使用: 角度ベースの重み付けで使用（等距離線生成とは無関係）
ノートブックを実行して、正しいマトリョーシカ形状の等距離線が描画されることを確認してください。