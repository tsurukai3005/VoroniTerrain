"""
ボクセルベース地形生成

主要機能:
- Voronoi分割による領域生成
- 迷路アルゴリズムによる通路/崖の配置
- 垂直な崖となめらかな坂の生成
- 距離ベースの高度差調整(0-40度の傾斜)
- パーリンノイズによる自然な凹凸
"""

import numpy as np
import plotly.graph_objects as go
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import random
from scipy.ndimage import gaussian_filter

# 日本語フォント設定(Windows環境)
plt.rcParams['font.family'] = ['Yu Gothic', 'MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========================================
# 地形生成パラメータ設定
# ========================================

class TerrainConfig:
    """地形生成のパラメータを一元管理するクラス"""

    # === マップサイズ設定 ===
    MAP_WIDTH = 400           # マップの幅(ピクセル)
    MAP_HEIGHT = 400          # マップの高さ(ピクセル)
    NUM_VORONOI_SITES = 400   # Voronoi分割の母点数(セル数)
    NUM_GROUPS = 60           # グループ数(領域数)

    # === Lloyd緩和設定 ===
    LLOYD_ITERATIONS = 3      # Lloyd緩和の反復回数(均等な点配置のため)

    # === 母点分布の疎密制御 ===
    CLUSTERING_STRENGTH = 0.3  # 0.0=均一分散, 1.0=強く密集
    CLUSTERING_SCALE = 100.0   # クラスタリングの空間スケール(px)

    # === 高度設定 ===
    INITIAL_HEIGHT = 40.0     # 初期高度(m)
    MIN_HEIGHT = 20.0         # 最低高度(m)
    MAX_HEIGHT = 80.0         # 最高高度(m)

    # === 基準点設定 ===
    # 高度の上昇/下降判定の基準点(左上中心)
    REFERENCE_POINT_X_RATIO = 0.25  # マップ幅の1/4の位置
    REFERENCE_POINT_Y_RATIO = 0.25  # マップ高さの1/4の位置

    # === 高度変化の確率設定 ===
    # 基準点に近づく場合の上昇確率(基準値)
    PROB_UP_APPROACHING = 0.75      # 80%で上昇
    # 基準点から離れる場合の上昇確率(基準値)
    PROB_UP_RECEDING = 0.25         # 20%で上昇

    # === 高度ベースの確率調整 ===
    HEIGHT_PROB_THRESHOLD_HIGH = 75.0  # この高度以上で下降しやすくなる(m)
    HEIGHT_PROB_THRESHOLD_LOW = 30.0   # この高度以下で上昇しやすくなる(m)
    HEIGHT_PROB_ADJUSTMENT = 0.30      # 確率調整の幅(±30%)

    # === 傾斜角度設定 ===
    MAX_SLOPE_ANGLE = 30.0    # 坂の最大傾斜角度(度)
    MIN_SLOPE_ANGLE = 0.0     # 坂の最小傾斜角度(度)

    # === 境界距離設定 ===
    FLAT_CENTER_DISTANCE = 30.0  # この距離以上離れた中央部は完全平坦(px)

    # === 崖(壁)の設定 ===
    CLIFF_RANGE = 5.0        # 崖の影響範囲(px) - 垂直に近い変化
    CLIFF_POWER = 0.1         # 崖の変化曲線(べき乗) - 小さいほど垂直に近い

    # === 坂(通路)の設定 ===
    SLOPE_RANGE = 50.0        # 坂の影響範囲(px) - 緩やかな変化
    SLOPE_POWER = 1.5         # 坂の変化曲線(べき乗) - 大きいほど緩やか

    # === パーリンノイズ設定 ===
    NOISE_AMPLITUDE = 8.0     # ノイズの振幅(m) - 細かな凹凸の高さ
    NOISE_OCTAVES = 4         # ノイズのオクターブ数 - 細かさのレベル
    NOISE_BASE_RES = 8        # ノイズの基本解像度
    NOISE_PERSISTENCE = 0.6   # オクターブ間の減衰率
    NOISE_BOUNDARY_FADE = 5.0  # 境界付近でノイズを減衰させる距離(px)
    NOISE_GAUSSIAN_SIGMA = 0.3  # ノイズ適用後の平滑化強度 (0.5→0.3で改善)

    # === 可視化設定 ===
    OUTPUT_DPI = 150          # 画像の解像度
    FIGURE_WIDTH_2D = 16      # 2D可視化の幅
    FIGURE_HEIGHT_2D = 8      # 2D可視化の高さ
    FIGURE_WIDTH_3D = 2000    # 3D可視化の幅
    FIGURE_HEIGHT_3D = 800    # 3D可視化の高さ

print("ライブラリのインポート完了")


def generate_perlin_noise_2d(shape, res, octaves=4, persistence=0.5):
    """
    2Dパーリンノイズを生成(改善版: 格子状アーティファクトを解消)

    Parameters:
    - shape: (height, width) 出力サイズ
    - res: (res_y, res_x) 基本周波数の解像度
    - octaves: オクターブ数(レイヤー数)
    - persistence: 各オクターブの振幅減衰率

    Returns:
    - noise: 正規化されたノイズ配列 [0, 1]
    """
    def fade(t):
        """スムーズステップ関数(5次エルミート補間)"""
        return 6*t**5 - 15*t**4 + 10*t**3

    def lerp(a, b, t):
        """線形補間"""
        return a + t * (b - a)

    noise = np.zeros(shape)
    amplitude = 1.0
    max_amplitude = 0.0

    for octave in range(octaves):
        # 現在のオクターブの周波数
        freq = 2 ** octave

        # グリッドサイズ(連続値として保持)
        grid_size_y = shape[0] / (res[0] * freq)
        grid_size_x = shape[1] / (res[1] * freq)

        # ランダムな勾配ベクトルテーブル
        np.random.seed(42 + octave)
        grid_h = int(res[0] * freq) + 2
        grid_w = int(res[1] * freq) + 2
        angles = 2 * np.pi * np.random.rand(grid_h, grid_w)
        grad_x = np.cos(angles)
        grad_y = np.sin(angles)

        # 各ピクセルの処理
        octave_noise = np.zeros(shape)
        for y in range(shape[0]):
            for x in range(shape[1]):
                # グリッド座標(浮動小数点)
                grid_y = y / grid_size_y
                grid_x = x / grid_size_x

                # グリッドセルの左上インデックス
                gy0 = int(np.floor(grid_y))
                gx0 = int(np.floor(grid_x))
                gy1 = gy0 + 1
                gx1 = gx0 + 1

                # グリッド内相対座標 [0, 1]
                ty = grid_y - gy0
                tx = grid_x - gx0

                # 4つの格子点の勾配ベクトル
                g00_x = grad_x[gy0 % grid_h, gx0 % grid_w]
                g00_y = grad_y[gy0 % grid_h, gx0 % grid_w]
                g10_x = grad_x[gy0 % grid_h, gx1 % grid_w]
                g10_y = grad_y[gy0 % grid_h, gx1 % grid_w]
                g01_x = grad_x[gy1 % grid_h, gx0 % grid_w]
                g01_y = grad_y[gy1 % grid_h, gx0 % grid_w]
                g11_x = grad_x[gy1 % grid_h, gx1 % grid_w]
                g11_y = grad_y[gy1 % grid_h, gx1 % grid_w]

                # 各格子点への距離ベクトル
                d00_x = tx
                d00_y = ty
                d10_x = tx - 1
                d10_y = ty
                d01_x = tx
                d01_y = ty - 1
                d11_x = tx - 1
                d11_y = ty - 1

                # ドット積
                n00 = g00_x * d00_x + g00_y * d00_y
                n10 = g10_x * d10_x + g10_y * d10_y
                n01 = g01_x * d01_x + g01_y * d01_y
                n11 = g11_x * d11_x + g11_y * d11_y

                # スムーズステップ補間
                fade_x = fade(tx)
                fade_y = fade(ty)

                # バイリニア補間
                n0 = lerp(n00, n10, fade_x)
                n1 = lerp(n01, n11, fade_x)
                value = lerp(n0, n1, fade_y)

                octave_noise[y, x] = value

        # オクターブを加算
        noise += amplitude * octave_noise
        max_amplitude += amplitude
        amplitude *= persistence

    # 正規化 [0, 1]
    noise = (noise + max_amplitude) / (2 * max_amplitude)
    return noise


class VoronoiTerrain:
    def __init__(self, width=None, height=None, num_sites=None, num_groups=None, config=None):
        """
        地形生成クラス

        Parameters:
        - width: マップ幅(デフォルト: TerrainConfig.MAP_WIDTH)
        - height: マップ高さ(デフォルト: TerrainConfig.MAP_HEIGHT)
        - num_sites: Voronoi母点数(デフォルト: TerrainConfig.NUM_VORONOI_SITES)
        - num_groups: グループ数(デフォルト: TerrainConfig.NUM_GROUPS)
        - config: 設定クラス(デフォルト: TerrainConfig)
        """
        self.config = config or TerrainConfig

        self.width = width or self.config.MAP_WIDTH
        self.height = height or self.config.MAP_HEIGHT
        self.num_sites = num_sites or self.config.NUM_VORONOI_SITES
        self.num_groups = num_groups or self.config.NUM_GROUPS

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

        # 等距離線ベースの高度マップ
        self.height_map = self._create_distance_based_height_map()

        # パーリンノイズで細かなディテールを追加
        self.height_map = self.apply_perlin_noise_detail(
            self.height_map,
            amplitude=self.config.NOISE_AMPLITUDE,
            octaves=self.config.NOISE_OCTAVES
        )

    def _generate_relaxed_points(self):
        """Lloyd緩和でより均等な点配置、または疎密制御による偏った配置

        CLUSTERING_STRENGTH = 0.0: 完全均一分散（Lloyd緩和）
        CLUSTERING_STRENGTH = 1.0: 強く密集した分布
        """
        clustering = self.config.CLUSTERING_STRENGTH

        if clustering <= 0.0:
            # 従来通りの均一分散（Lloyd緩和）
            points = np.random.rand(self.num_sites, 2)
            points[:, 0] *= self.width
            points[:, 1] *= self.height

            for _ in range(self.config.LLOYD_ITERATIONS):
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
        else:
            # 疎密制御による分布
            # パーリンノイズで密度マップを生成
            density_res = max(2, int(min(self.width, self.height) / self.config.CLUSTERING_SCALE))
            density_map = generate_perlin_noise_2d(
                shape=(self.height, self.width),
                res=(density_res, density_res),
                octaves=3,
                persistence=0.5
            )

            # 密度マップを0-1に正規化
            density_map = (density_map - density_map.min()) / (density_map.max() - density_map.min())

            # clustering strengthで均一分布と密度ベース分布を混合
            uniform_points = np.random.rand(self.num_sites, 2)
            uniform_points[:, 0] *= self.width
            uniform_points[:, 1] *= self.height

            # 密度ベースのポイント生成（rejection sampling）
            clustered_points = []
            max_attempts = self.num_sites * 100
            attempts = 0

            while len(clustered_points) < self.num_sites and attempts < max_attempts:
                x = int(np.random.rand() * self.width)
                y = int(np.random.rand() * self.height)
                x = np.clip(x, 0, self.width - 1)
                y = np.clip(y, 0, self.height - 1)

                # 密度マップの値に基づいて受理
                if np.random.rand() < density_map[y, x]:
                    clustered_points.append([x, y])

                attempts += 1

            # 不足分は均一分布で補完
            while len(clustered_points) < self.num_sites:
                x = np.random.rand() * self.width
                y = np.random.rand() * self.height
                clustered_points.append([x, y])

            clustered_points = np.array(clustered_points[:self.num_sites])

            # clustering strengthで混合
            points = (1.0 - clustering) * uniform_points + clustering * clustered_points

            # Lloyd緩和を弱めに適用（clustering強度に応じて）
            lloyd_iters = int(self.config.LLOYD_ITERATIONS * (1.0 - clustering))
            for _ in range(lloyd_iters):
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
        """セルをグループ化(母点数で小:中:大 = 1:3:9、出現確率均等)"""
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
                target_sizes.append(int(cells_per_group * 5.0))
            else:
                target_sizes.append(int(cells_per_group * 25.0))

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

    def _calculate_boundary_lengths(self):
        """各グループ間の境界線の長さを計算"""
        boundary_lengths = {}

        for y in range(self.height):
            for x in range(self.width):
                current = self.group_map[y, x]

                # 右隣をチェック
                if x < self.width - 1:
                    neighbor = self.group_map[y, x + 1]
                    if neighbor != current:
                        key = tuple(sorted([current, neighbor]))
                        boundary_lengths[key] = boundary_lengths.get(key, 0) + 1

                # 下隣をチェック
                if y < self.height - 1:
                    neighbor = self.group_map[y + 1, x]
                    if neighbor != current:
                        key = tuple(sorted([current, neighbor]))
                        boundary_lengths[key] = boundary_lengths.get(key, 0) + 1

        return boundary_lengths

    def _generate_maze(self):
        """グループ間で迷路を生成(通路と壁) - より開けた構造

        v11改善: 境界線の長さを考慮し、長い境界を優先的に通路にする
        """
        # 境界線の長さを事前計算
        boundary_lengths = self._calculate_boundary_lengths()

        visited = {0}
        # エッジに優先度(境界線の長さ)を付与
        edges = [(0, neighbor, boundary_lengths.get(tuple(sorted([0, neighbor])), 0))
                 for neighbor in self.group_edges[0]]
        self.is_wall = {}

        while len(visited) < self.num_groups and edges:
            # 境界線が長いエッジを優先的に選択(重み付きランダム選択)
            # 長さの合計を計算
            total_length = sum(length for _, _, length in edges)

            if total_length > 0:
                # 長さに比例した確率で選択
                rand_val = random.random() * total_length
                cumulative = 0
                selected_idx = 0

                for i, (_, _, length) in enumerate(edges):
                    cumulative += length
                    if rand_val <= cumulative:
                        selected_idx = i
                        break
            else:
                # 長さ情報がない場合はランダム
                selected_idx = random.randint(0, len(edges) - 1)

            from_group, to_group, _ = edges.pop(selected_idx)

            if to_group not in visited:
                visited.add(to_group)
                key = tuple(sorted([from_group, to_group]))
                self.is_wall[key] = False

                for neighbor in self.group_edges[to_group]:
                    if neighbor not in visited:
                        edge_length = boundary_lengths.get(tuple(sorted([to_group, neighbor])), 0)
                        edges.append((to_group, neighbor, edge_length))

        # 残りのエッジを壁として設定
        for g1 in range(self.num_groups):
            for g2 in self.group_edges[g1]:
                key = tuple(sorted([g1, g2]))
                if key not in self.is_wall:
                    self.is_wall[key] = True

        # v11改善: 長い境界を通路に優先割り当て

    def _compute_group_centroids(self):
        """各グループの重心座標を計算"""
        group_centroids = {}

        for group_id, cells in self.group_cells.items():
            centroid_x = np.mean([self.points[cell_id][0] for cell_id in cells])
            centroid_y = np.mean([self.points[cell_id][1] for cell_id in cells])
            group_centroids[group_id] = np.array([centroid_x, centroid_y])

        return group_centroids

    def _adjust_probability_by_height(self, base_prob, current_height):
        """現在の高度に基づいて上昇確率を調整

        Parameters:
        - base_prob: 基準となる上昇確率
        - current_height: 現在の高度(m)

        Returns:
        - adjusted_prob: 調整後の上昇確率
        """
        # 高すぎる場合は上昇確率を下げる
        if current_height >= self.config.HEIGHT_PROB_THRESHOLD_HIGH:
            # 75m以上: 下降しやすくする(上昇確率を減らす)
            adjustment = -self.config.HEIGHT_PROB_ADJUSTMENT
        # 低すぎる場合は上昇確率を上げる
        elif current_height <= self.config.HEIGHT_PROB_THRESHOLD_LOW:
            # 25m以下: 上昇しやすくする(上昇確率を増やす)
            adjustment = self.config.HEIGHT_PROB_ADJUSTMENT
        else:
            # 中間の高度では調整なし
            adjustment = 0.0

        # 確率を調整(0.0-1.0の範囲にクリップ)
        adjusted_prob = np.clip(base_prob + adjustment, 0.0, 1.0)
        return adjusted_prob

    def _propagate_heights(self):
        """グループごとに高度を伝播(基準点への距離に基づく方向決定)

        改善点:
        - 領域の中心間距離に基づいて高度差を設定(0-40度の範囲)
        - 通路で繋がった隣接領域の高度差を距離ベースで調整
        """
        # 基準点: 左上の中心
        self.target_point = np.array([
            self.width * self.config.REFERENCE_POINT_X_RATIO,
            self.height * self.config.REFERENCE_POINT_Y_RATIO
        ])

        # 各グループの重心を計算
        group_centroids = self._compute_group_centroids()

        heights = {}
        heights[0] = self.config.INITIAL_HEIGHT

        visited = {0}
        queue = deque([0])

        while queue:
            current = queue.popleft()

            # 現在のノードから未訪問の隣接ノード(通路)を取得
            unvisited_neighbors = []
            for neighbor in self.group_edges[current]:
                key = tuple(sorted([current, neighbor]))
                if not self.is_wall[key] and neighbor not in visited:
                    unvisited_neighbors.append(neighbor)

            # 現在の基準点からの距離
            current_dist = np.linalg.norm(group_centroids[current] - self.target_point)

            # 分岐の場合
            if len(unvisited_neighbors) > 1:
                # 各隣接ノードの基準点からの距離を計算し、近い/遠いで分類
                neighbor_dists = [(n, np.linalg.norm(group_centroids[n] - self.target_point))
                                 for n in unvisited_neighbors]

                # 距離でソート
                neighbor_dists.sort(key=lambda x: x[1])

                # 近い方は上昇、遠い方は下降の傾向
                for i, (neighbor, neighbor_dist) in enumerate(neighbor_dists):
                    visited.add(neighbor)

                    # 基準点に近づく/離れる場合の基準確率
                    if neighbor_dist < current_dist:
                        # 基準点に近づく → 上昇しやすい
                        base_prob = self.config.PROB_UP_APPROACHING
                    else:
                        # 基準点から離れる → 下降しやすい
                        base_prob = self.config.PROB_UP_RECEDING

                    # 現在の高度に基づいて確率を調整
                    current_height = heights[current]
                    adjusted_prob = self._adjust_probability_by_height(base_prob, current_height)

                    # 調整された確率で上昇/下降を判定
                    go_up = random.random() < adjusted_prob

                    # 領域中心間の距離を計算
                    centroid_distance = np.linalg.norm(
                        group_centroids[current] - group_centroids[neighbor]
                    )

                    # 距離に基づいて高度差を計算(MIN_SLOPE_ANGLE-MAX_SLOPE_ANGLE度の範囲)
                    # tan(angle) = height_diff / distance
                    # ランダムな角度を設定範囲内で選択
                    target_angle_deg = random.uniform(
                        self.config.MIN_SLOPE_ANGLE,
                        self.config.MAX_SLOPE_ANGLE
                    )
                    target_angle_rad = np.radians(target_angle_deg)
                    height_diff = centroid_distance * np.tan(target_angle_rad)

                    if go_up:
                        heights[neighbor] = min(self.config.MAX_HEIGHT, heights[current] + height_diff)
                    else:
                        heights[neighbor] = max(self.config.MIN_HEIGHT, heights[current] - height_diff)

                    queue.append(neighbor)

            # 分岐なし(1つの隣接ノード)
            elif len(unvisited_neighbors) == 1:
                neighbor = unvisited_neighbors[0]
                visited.add(neighbor)

                # 隣接ノードの基準点からの距離
                neighbor_dist = np.linalg.norm(group_centroids[neighbor] - self.target_point)

                # 基準点に近づく/離れる場合の基準確率
                if neighbor_dist < current_dist:
                    # 基準点に近づく → 上昇しやすい
                    base_prob = self.config.PROB_UP_APPROACHING
                else:
                    # 基準点から離れる → 下降しやすい
                    base_prob = self.config.PROB_UP_RECEDING

                # 現在の高度に基づいて確率を調整
                current_height = heights[current]
                adjusted_prob = self._adjust_probability_by_height(base_prob, current_height)

                # 調整された確率で上昇/下降を判定
                go_up = random.random() < adjusted_prob

                # 領域中心間の距離を計算
                centroid_distance = np.linalg.norm(
                    group_centroids[current] - group_centroids[neighbor]
                )

                # 距離に基づいて高度差を計算(MIN_SLOPE_ANGLE-MAX_SLOPE_ANGLE度の範囲)
                target_angle_deg = random.uniform(
                    self.config.MIN_SLOPE_ANGLE,
                    self.config.MAX_SLOPE_ANGLE
                )
                target_angle_rad = np.radians(target_angle_deg)
                height_diff = centroid_distance * np.tan(target_angle_rad)

                if go_up:
                    heights[neighbor] = min(self.config.MAX_HEIGHT, heights[current] + height_diff)
                else:
                    heights[neighbor] = max(self.config.MIN_HEIGHT, heights[current] - height_diff)

                queue.append(neighbor)

        # 未訪問のグループに平均高度を割り当て
        avg_height = np.mean(list(heights.values()))
        for i in range(self.num_groups):
            if i not in heights:
                heights[i] = avg_height

        # 通路で繋がった隣接領域の高度差を検証・調整(距離ベース)
        for key, is_wall in self.is_wall.items():
            if not is_wall:  # 通路(坂)の場合のみ
                g1, g2 = key
                if g1 in heights and g2 in heights:
                    # 領域中心間の距離を計算
                    centroid_distance = np.linalg.norm(
                        group_centroids[g1] - group_centroids[g2]
                    )

                    # 現在の高度差から傾斜角度を計算
                    current_height_diff = abs(heights[g1] - heights[g2])
                    current_angle_rad = np.arctan(current_height_diff / centroid_distance)
                    current_angle_deg = np.degrees(current_angle_rad)

                    # MAX_SLOPE_ANGLEを超える場合は調整
                    if current_angle_deg > self.config.MAX_SLOPE_ANGLE:
                        # MAX_SLOPE_ANGLEに対応する高度差を計算
                        max_height_diff = centroid_distance * np.tan(
                            np.radians(self.config.MAX_SLOPE_ANGLE)
                        )

                        # 高度差を調整(比率を保ちながら縮小)
                        avg = (heights[g1] + heights[g2]) / 2.0
                        adjustment = max_height_diff / 2.0

                        if heights[g1] > heights[g2]:
                            heights[g1] = avg + adjustment
                            heights[g2] = avg - adjustment
                        else:
                            heights[g1] = avg - adjustment
                            heights[g2] = avg + adjustment

        return heights

    def _compute_boundary_pixels_and_distance_field(self):
        """境界ピクセルの検出と距離フィールドの計算(座標系を統一)"""
        self.group_boundaries = {i: {} for i in range(self.num_groups)}

        # 境界ピクセルを検出((x, y)形式で保存)
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
                            # (x, y)形式で保存
                            self.group_boundaries[current_group][neighbor_group].append((x, y))

        # 距離フィールドを計算
        self.distance_fields = {}
        self.distance_to_boundary = {}

        for group_id in range(self.num_groups):
            dist_field = {}
            all_boundary_pixels = []

            for neighbor_id, boundary_pixels in self.group_boundaries[group_id].items():
                all_boundary_pixels.extend(boundary_pixels)

                if len(boundary_pixels) == 0:
                    continue

                boundary_array = np.array(boundary_pixels)
                dist_field[neighbor_id] = {}

                for y in range(self.height):
                    for x in range(self.width):
                        if self.group_map[y, x] != group_id:
                            continue

                        dists = np.sqrt((boundary_array[:, 0] - x)**2 +
                                       (boundary_array[:, 1] - y)**2)
                        min_dist = np.min(dists)
                        dist_field[neighbor_id][(x, y)] = min_dist

            self.distance_fields[group_id] = dist_field

            # 全境界への最小距離
            if len(all_boundary_pixels) > 0:
                all_boundary_array = np.array(all_boundary_pixels)
                dist_map = {}

                for y in range(self.height):
                    for x in range(self.width):
                        if self.group_map[y, x] != group_id:
                            continue

                        dists = np.sqrt((all_boundary_array[:, 0] - x)**2 +
                                       (all_boundary_array[:, 1] - y)**2)
                        min_dist = np.min(dists)
                        dist_map[(x, y)] = min_dist

                self.distance_to_boundary[group_id] = dist_map

    def _compute_boundary_centroids(self):
        """各境界の中央点(境界線上の中点)を計算"""
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

    def _create_distance_based_height_map(self):
        """等距離線ベースの高度マップ生成(改善版v2: 上り→下り境界の滑らかな接続)

        アルゴリズム:
        1. 各坂境界を「上り」「下り」に分類
        2. 上り境界から下り境界への滑らかな遷移を実装
        3. 崖は垂直方向に急激に変化
        4. 境界間の補間で段差を解消
        """
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

                # 境界までの最小距離を取得
                dist_to_any_boundary = self.distance_to_boundary.get(current_group, {}).get((x, y), float('inf'))

                # 中央平坦化: 境界から一定距離以上離れている場合は完全平坦
                if dist_to_any_boundary >= self.config.FLAT_CENTER_DISTANCE:
                    height_map[y, x] = base_height
                    continue

                # 各境界への情報を取得
                dist_field = self.distance_fields.get(current_group, {})
                if not dist_field:
                    height_map[y, x] = base_height
                    continue

                # === ステップ1: 坂境界を分類 ===
                uphill_boundaries = []    # (neighbor_id, boundary_center, dist)
                downhill_boundaries = []  # (neighbor_id, boundary_center, dist)
                cliff_boundaries = []     # (neighbor_id, boundary_center, dist, neighbor_height)

                for neighbor_id, pixel_distances in dist_field.items():
                    dist_to_this_boundary = pixel_distances.get((x, y), float('inf'))

                    if dist_to_this_boundary == float('inf'):
                        continue

                    key = tuple(sorted([current_group, neighbor_id]))
                    if key not in self.boundary_centroids:
                        continue

                    boundary_center = self.boundary_centroids[key]
                    is_wall = self.is_wall.get(key, True)
                    neighbor_height = self.group_heights[neighbor_id]

                    if is_wall:
                        # 崖(後で処理)
                        cliff_boundaries.append((neighbor_id, boundary_center, dist_to_this_boundary, neighbor_height))
                    else:
                        # 坂: 上り/下りを判定
                        if neighbor_height > base_height:
                            uphill_boundaries.append((neighbor_id, boundary_center, dist_to_this_boundary))
                        else:
                            downhill_boundaries.append((neighbor_id, boundary_center, dist_to_this_boundary))

                # === ステップ2: 坂の影響を先に計算 ===
                slope_height = base_height  # デフォルトはベース高度

                if len(uphill_boundaries) > 0 or len(downhill_boundaries) > 0:
                    # 最も近い上り境界と下り境界を取得
                    nearest_uphill = None
                    nearest_downhill = None

                    if len(uphill_boundaries) > 0:
                        uphill_boundaries.sort(key=lambda item: item[2])
                        nearest_uphill = uphill_boundaries[0]

                    if len(downhill_boundaries) > 0:
                        downhill_boundaries.sort(key=lambda item: item[2])
                        nearest_downhill = downhill_boundaries[0]

                    # ケース1: 上り境界のみ
                    if nearest_uphill and not nearest_downhill:
                        neighbor_id, _, dist = nearest_uphill
                        neighbor_height = self.group_heights[neighbor_id]

                        if dist < self.config.SLOPE_RANGE:
                            t = dist / self.config.SLOPE_RANGE
                            influence = 1.0 - (t ** self.config.SLOPE_POWER)
                            target_height = (base_height + neighbor_height) / 2.0
                            slope_height = base_height * (1.0 - influence) + target_height * influence
                        else:
                            slope_height = base_height

                    # ケース2: 下り境界のみ
                    elif nearest_downhill and not nearest_uphill:
                        neighbor_id, _, dist = nearest_downhill
                        neighbor_height = self.group_heights[neighbor_id]

                        if dist < self.config.SLOPE_RANGE:
                            t = dist / self.config.SLOPE_RANGE
                            influence = 1.0 - (t ** self.config.SLOPE_POWER)
                            target_height = (base_height + neighbor_height) / 2.0
                            slope_height = base_height * (1.0 - influence) + target_height * influence
                        else:
                            slope_height = base_height

                    # ケース3: 上り境界と下り境界の両方あり
                    else:
                        uphill_id, _, uphill_dist = nearest_uphill
                        downhill_id, _, downhill_dist = nearest_downhill

                        uphill_height = self.group_heights[uphill_id]
                        downhill_height = self.group_heights[downhill_id]

                        # 上り境界と下り境界への距離の比率で補間
                        total_dist = uphill_dist + downhill_dist

                        if total_dist >= 0.1:
                            # 距離の逆数で重み付け(近いほど影響大)
                            uphill_weight = (1.0 / (uphill_dist + 1.0))
                            downhill_weight = (1.0 / (downhill_dist + 1.0))
                            total_weight = uphill_weight + downhill_weight

                            # 上り境界への目標高度(中間高度)
                            uphill_target = (base_height + uphill_height) / 2.0
                            # 下り境界への目標高度(中間高度)
                            downhill_target = (base_height + downhill_height) / 2.0

                            # 距離に基づく影響度
                            uphill_influence = 0.0
                            downhill_influence = 0.0

                            if uphill_dist < self.config.SLOPE_RANGE:
                                t = uphill_dist / self.config.SLOPE_RANGE
                                uphill_influence = 1.0 - (t ** self.config.SLOPE_POWER)

                            if downhill_dist < self.config.SLOPE_RANGE:
                                t = downhill_dist / self.config.SLOPE_RANGE
                                downhill_influence = 1.0 - (t ** self.config.SLOPE_POWER)

                            # 重み付き平均で高度を計算
                            if uphill_influence + downhill_influence > 0.0:
                                # 上り境界と下り境界の影響を滑らかに混合
                                blended_height = (
                                    uphill_target * uphill_influence * (uphill_weight / total_weight) +
                                    downhill_target * downhill_influence * (downhill_weight / total_weight)
                                ) / (uphill_influence * (uphill_weight / total_weight) +
                                     downhill_influence * (downhill_weight / total_weight))

                                # ベース高度と混合
                                total_influence = min(1.0, uphill_influence + downhill_influence)
                                slope_height = base_height * (1.0 - total_influence) + blended_height * total_influence

                # === ステップ3: 崖の影響を適用(坂で傾斜した後の高度を基準) ===
                if len(cliff_boundaries) > 0:
                    # 現在位置の高度(坂の影響込み)
                    current_height = slope_height

                    # 最も影響の強い崖を探す
                    max_cliff_influence = 0.0
                    cliff_target_height = current_height

                    for neighbor_id, _, dist, neighbor_base_height in cliff_boundaries:
                        if dist >= self.config.CLIFF_RANGE:
                            continue

                        # 境界での隣接領域の高度を推定
                        # (隣接領域も坂の影響を受けていると仮定)
                        # 簡略化: 隣接領域の基準高度を使用
                        neighbor_boundary_height = neighbor_base_height

                        # 実際の高度差で上下を判定
                        height_diff = current_height - neighbor_boundary_height

                        # 高い側から低い側への崖のみ作成
                        if height_diff > 0:
                            t = dist / self.config.CLIFF_RANGE
                            influence = 1.0 - (t ** self.config.CLIFF_POWER)

                            if influence > max_cliff_influence:
                                max_cliff_influence = influence
                                cliff_target_height = neighbor_boundary_height

                    # 崖の影響を適用
                    if max_cliff_influence > 0.0:
                        height_map[y, x] = current_height * (1.0 - max_cliff_influence) + cliff_target_height * max_cliff_influence
                    else:
                        height_map[y, x] = slope_height
                else:
                    # 崖がない場合は坂の高度をそのまま使用
                    height_map[y, x] = slope_height

        return height_map

    def apply_perlin_noise_detail(self, height_map, amplitude, octaves):
        """
        高度マップにパーリンノイズによる細かなディテールを追加

        Parameters:
        - height_map: 元の高度マップ
        - amplitude: ノイズの振幅(メートル)
        - octaves: ノイズのオクターブ数(細かさ)

        Returns:
        - height_map: ノイズ適用後の高度マップ
        """
        # パーリンノイズ生成
        noise = generate_perlin_noise_2d(
            shape=(self.height, self.width),
            res=(self.config.NOISE_BASE_RES, self.config.NOISE_BASE_RES),
            octaves=octaves,
            persistence=self.config.NOISE_PERSISTENCE
        )

        # ノイズを [-amplitude, amplitude] の範囲にスケール
        noise = (noise - 0.5) * 2.0 * amplitude

        # 境界付近ではノイズを弱める
        for y in range(self.height):
            for x in range(self.width):
                current_group = self.group_map[y, x]

                # 境界までの最小距離を取得
                dist_to_boundary = self.distance_to_boundary.get(current_group, {}).get((x, y), 0.0)

                # 境界から一定距離以内ではノイズを減衰
                if dist_to_boundary < self.config.NOISE_BOUNDARY_FADE:
                    noise_strength = dist_to_boundary / self.config.NOISE_BOUNDARY_FADE
                    noise[y, x] *= noise_strength

        # ノイズを適用
        height_map_with_noise = height_map + noise

        # ガウシアンフィルタで軽く滑らかに
        height_map_with_noise = gaussian_filter(height_map_with_noise, sigma=self.config.NOISE_GAUSSIAN_SIGMA)

        return height_map_with_noise


def calculate_slope_angles(height_map):
    """各ピクセルの傾斜角度を計算"""
    dy, dx = np.gradient(height_map)
    slope = np.sqrt(dx**2 + dy**2)
    angle = np.arctan(slope) * 180 / np.pi  # 度に変換
    return angle


def visualize_2d(terrain):
    """2D可視化"""
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

    # グループ構造の可視化
    fig, axes = plt.subplots(1, 2, figsize=(TerrainConfig.FIGURE_WIDTH_2D, TerrainConfig.FIGURE_HEIGHT_2D))

    # === 左: グループマップ(高さと境界種別を表示)===
    ax1 = axes[0]

    # 各グループの高さで色分け
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

            for dy, dx in [(0, 1), (1, 0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < terrain.height and 0 <= nx < terrain.width:
                    neighbor_group = terrain.group_map[ny, nx]
                    if neighbor_group != current_group:
                        key = tuple(sorted([current_group, neighbor_group]))
                        is_wall = terrain.is_wall.get(key, False)
                        segment_type = 'wall' if is_wall else 'passage'
                        boundary_segments[segment_type].append(([x, nx], [y, ny]))

    # 境界を描画
    for segment_type, color in [('wall', 'red'), ('passage', 'green')]:
        for xs, ys in boundary_segments[segment_type]:
            ax1.plot(xs, ys, color=color, linewidth=2.5, alpha=0.9, solid_capstyle='round')

    # 重心に点を描画
    for group_id, centroid in group_centroids.items():
        ax1.plot(centroid[0], centroid[1], 'ko', markersize=4)

    # 通路方向への矢印
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

            dx1 = boundary_center[0] - c1[0]
            dy1 = boundary_center[1] - c1[1]
            ax1.arrow(c1[0], c1[1], dx1*0.8, dy1*0.8,
                     head_width=2, head_length=2, fc=color, ec=color, alpha=arrow_alpha, linewidth=1.5)

            dx2 = boundary_center[0] - c2[0]
            dy2 = boundary_center[1] - c2[1]
            ax1.arrow(c2[0], c2[1], dx2*0.8, dy2*0.8,
                     head_width=2, head_length=2, fc=color, ec=color, alpha=arrow_alpha, linewidth=1.5)

    # v13: 基準点を赤点で表示(左側)
    if hasattr(terrain, 'target_point'):
        ax1.plot(terrain.target_point[0], terrain.target_point[1], 'r*', markersize=20,
                label='基準点(高度上昇の目標)', zorder=10, markeredgecolor='white', markeredgewidth=1.5)
        ax1.legend(loc='upper right')

    ax1.set_title('グループ分割(坂の方向を矢印で表示)\n緑=通路(緩やかな坂)、赤=壁(崖)', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, label='高度 (m)')

    # === 右: 高度マップ ===
    ax2 = axes[1]
    im2 = ax2.imshow(terrain.height_map, cmap=cmap_blue_orange, interpolation='nearest')

    # 境界を描画
    for xs, ys in boundary_segments['wall'] + boundary_segments['passage']:
        ax2.plot(xs, ys, color='black', linewidth=2.0, alpha=1.0, solid_capstyle='round')

    # 等距離線を描画
    for group_id in range(terrain.num_groups):
        dist_to_boundary = terrain.distance_to_boundary.get(group_id, {})

        if len(dist_to_boundary) == 0:
            continue

        dist_map = np.full((terrain.height, terrain.width), np.nan)
        for (px, py), dist in dist_to_boundary.items():
            dist_map[py, px] = dist

        x_grid = np.arange(terrain.width)
        y_grid = np.arange(terrain.height)

        for target_dist in [10, 20, 30]:
            valid_mask = ~np.isnan(dist_map)

            if valid_mask.sum() > 10:
                try:
                    ax2.contour(x_grid, y_grid, dist_map,
                               levels=[target_dist],
                               colors='black',
                               linewidths=0.5,
                               alpha=0.4)
                except:
                    pass

    # 重心と境界中央を結ぶ線
    for group_id in range(terrain.num_groups):
        if group_id not in group_centroids:
            continue

        centroid = group_centroids[group_id]

        for neighbor_id in terrain.group_boundaries[group_id].keys():
            key = tuple(sorted([group_id, neighbor_id]))
            if key not in terrain.boundary_centroids:
                continue

            boundary_center = terrain.boundary_centroids[key]

            ax2.plot([centroid[0], boundary_center[0]], [centroid[1], boundary_center[1]],
                    'b--', linewidth=1, alpha=0.6)

            if neighbor_id in group_centroids:
                neighbor_centroid = group_centroids[neighbor_id]
                ax2.plot([boundary_center[0], neighbor_centroid[0]],
                        [boundary_center[1], neighbor_centroid[1]],
                        'b--', linewidth=1, alpha=0.6)

    # v13: 基準点を赤点で表示
    if hasattr(terrain, 'target_point'):
        ax2.plot(terrain.target_point[0], terrain.target_point[1], 'r*', markersize=20,
                label='基準点(高度上昇の目標)', zorder=10, markeredgecolor='white', markeredgewidth=1.5)
        ax2.legend(loc='upper right')

    ax2.set_title('高度マップ(境界=黒太線、等距離線=黒細線、重心↔境界中央=青破線)', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    plt.colorbar(im2, ax=ax2, label='高度 (m)')

    plt.tight_layout()
    plt.savefig('terrain_2d.png', dpi=TerrainConfig.OUTPUT_DPI, bbox_inches='tight')
    print("2D可視化を保存: terrain_2d.png")
    plt.show()


def visualize_3d(terrain):
    """3D可視化"""
    blue_orange_scale = [
        [0.0, 'rgb(0, 102, 204)'],
        [0.2, 'rgb(51, 153, 255)'],
        [0.4, 'rgb(153, 204, 255)'],
        [0.6, 'rgb(255, 204, 153)'],
        [0.8, 'rgb(255, 153, 51)'],
        [1.0, 'rgb(255, 102, 0)']
    ]

    height, width = terrain.height_map.shape
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    X, Y = np.meshgrid(x, y)
    Z = terrain.height_map

    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=blue_orange_scale,
        colorbar=dict(title="Height (m)"),
        hovertemplate='X: %{x}<br>Y: %{y}<br>Height: %{z:.1f}m<extra></extra>'
    )])

    # 高度の範囲を取得して適切なスケールを計算
    height_range = terrain.height_map.max() - terrain.height_map.min()

    # 水平方向のスケールに合わせてZ軸の比率を計算
    # マップサイズ(400x400)に対する高度範囲の比率
    z_ratio = height_range / terrain.width

    fig.update_layout(
        title='Surface Mesh Terrain (Interactive 3D)',
        scene=dict(
            xaxis_title='X (px)',
            yaxis_title='Y (px)',
            zaxis_title='Height (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z_ratio),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        width=TerrainConfig.FIGURE_WIDTH_3D,
        height=TerrainConfig.FIGURE_HEIGHT_3D
    )

    fig.write_html('terrain_3d.html')
    print("3D可視化を保存: terrain_3d.html")
    fig.show()


def print_statistics(terrain):
    """統計情報を出力"""
    # グループサイズ統計
    group_sizes = {i: len(cells) for i, cells in terrain.group_cells.items()}
    print("\n=== グループサイズ統計 ===")
    sizes = sorted(group_sizes.values(), reverse=True)
    print(f"最大3グループ: {sizes[:3]}")
    print(f"最小3グループ: {sizes[-3:]}")
    print(f"サイズ比(最大/最小): {sizes[0] / sizes[-1]:.2f}")

    # 高度統計
    print("\n=== 高度統計 ===")
    print(f"最低高度: {terrain.height_map.min():.1f}m")
    print(f"最高高度: {terrain.height_map.max():.1f}m")
    print(f"平均高度: {terrain.height_map.mean():.1f}m")
    print(f"高度差: {terrain.height_map.max() - terrain.height_map.min():.1f}m")

    # 壁と通路の統計
    num_walls = sum(1 for is_wall in terrain.is_wall.values() if is_wall)
    num_passages = len(terrain.is_wall) - num_walls
    print("\n=== 構造統計 ===")
    print(f"壁の数: {num_walls}")
    print(f"通路の数: {num_passages}")
    print(f"通路比率: {num_passages / len(terrain.is_wall) * 100:.1f}%")

    # 通路の高度差統計
    passage_height_diffs = []
    for key, is_wall in terrain.is_wall.items():
        if not is_wall:
            g1, g2 = key
            diff = abs(terrain.group_heights[g1] - terrain.group_heights[g2])
            passage_height_diffs.append(diff)

    if passage_height_diffs:
        print("\n=== 通路の高度差(隣接グループ間) ===")
        print(f"最小: {min(passage_height_diffs):.1f}m")
        print(f"最大: {max(passage_height_diffs):.1f}m")
        print(f"平均: {np.mean(passage_height_diffs):.1f}m")
        print(f"目標範囲: 1-2.5m")

    # 壁の高度差統計
    wall_height_diffs = []
    for key, is_wall in terrain.is_wall.items():
        if is_wall:
            g1, g2 = key
            diff = abs(terrain.group_heights[g1] - terrain.group_heights[g2])
            wall_height_diffs.append(diff)

    if wall_height_diffs:
        print("\n=== 壁の高度差(非隣接グループ間) ===")
        print(f"最小: {min(wall_height_diffs):.1f}m")
        print(f"最大: {max(wall_height_diffs):.1f}m")
        print(f"平均: {np.mean(wall_height_diffs):.1f}m")
        print(f"目標範囲: 20-30m")

    # 傾斜角度統計
    angles = calculate_slope_angles(terrain.height_map)

    flat = np.sum(angles < 10)
    gentle = np.sum((angles >= 10) & (angles < 40))
    steep = np.sum(angles >= 40)
    total = angles.size

    print(f"\n=== 傾斜角度統計 ===")
    print(f"平地(0-10度): {flat} ピクセル ({flat/total*100:.1f}%)")
    print(f"緩やか(10-40度): {gentle} ピクセル ({gentle/total*100:.1f}%)")
    print(f"急峻(40度以上): {steep} ピクセル ({steep/total*100:.1f}%)")
    print(f"最大傾斜角度: {angles.max():.1f}度")
    print(f"平均傾斜角度: {angles.mean():.1f}度")

    moderate = np.sum((angles >= 10) & (angles < 20))
    print(f"\n10-20度の緩やかな坂: {moderate} ピクセル ({moderate/total*100:.1f}%)")


def visualize_slope_angles(terrain):
    """傾斜角度の可視化"""
    angles = calculate_slope_angles(terrain.height_map)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(angles, cmap='RdYlGn_r', vmin=0, vmax=60)
    plt.colorbar(label='傾斜角度 (度)')
    plt.title('傾斜角度マップ')
    plt.xlabel('X')
    plt.ylabel('Y')

    plt.subplot(1, 2, 2)
    plt.hist(angles.flatten(), bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=10, color='b', linestyle='--', label='10度(平地の上限)')
    plt.axvline(x=20, color='g', linestyle='--', label='20度(緩やかな坂の上限目標)')
    plt.axvline(x=40, color='orange', linestyle='--', label='40度(中程度の坂)')
    plt.axvline(x=50, color='r', linestyle='--', label='50度(急峻な坂)')
    plt.xlabel('傾斜角度 (度)')
    plt.ylabel('ピクセル数')
    plt.title('傾斜角度分布')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('terrain_slope_angles.png', dpi=TerrainConfig.OUTPUT_DPI, bbox_inches='tight')
    print("傾斜角度可視化を保存: terrain_slope_angles.png")
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("ボクセルベース地形生成")
    print("=" * 60)

    # 地形を生成
    print("\n地形を生成中...")
    terrain = VoronoiTerrain()

    print(f"\n生成完了:")
    print(f"  - セル数: {terrain.num_sites}")
    print(f"  - グループ数: {terrain.num_groups}")
    print(f"  - 地形サイズ: {terrain.width}x{terrain.height}")

    # 統計情報を出力
    print_statistics(terrain)

    # 可視化
    print("\n可視化を生成中...")
    visualize_2d(terrain)
    visualize_slope_angles(terrain)
    visualize_3d(terrain)

    print("\n" + "=" * 60)
    print("完了!生成されたファイル:")
    print("  - terrain_2d.png (2D可視化)")
    print("  - terrain_slope_angles.png (傾斜角度)")
    print("  - terrain_3d.html (インタラクティブ3D)")
    print("=" * 60)
