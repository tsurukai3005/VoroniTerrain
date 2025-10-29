# 地形生成アルゴリズム改善案 v4

## 問題点

現在の実装では:
1. 境界付近でのみ高度が変化し、中央部が平坦
2. 方向判定により鋭利な高度差が中央付近に発生
3. 等距離線が等高線として機能していない

## 改善アルゴリズム

### 基本方針

各ピクセルの高度は、そのピクセルから見た全方向の境界への寄与を統合して決定する。

### ステップ1: 各境界方向への寄与計算

各ピクセル `(x, y)` について、各境界 `b` に対して:

1. **距離**: 境界 `b` までの最短距離 `d_b`
2. **方向角度**: 重心から境界中央への方向と、重心からピクセルへの方向の角度差 `θ_b`
3. **境界タイプ**: 壁 or 通路

### ステップ2: 方向ごとの高度ブレンド計算

各境界 `b` について:

```python
# 境界からの距離を正規化（0=境界, 1=中心）
# distance_to_boundary_map から最大距離を取得
max_distance = max(distance_to_boundary_map.values())
normalized_distance = d_b / max_distance  # 0-1の範囲

# 境界bの影響力を計算（等距離線に沿った高度変化）
if 境界bが壁（崖）:
    # 崖: 境界付近(0-0.3)で急激に変化、それ以外(0.3-1.0)はほぼ平坦
    if normalized_distance < 0.3:
        # 境界付近: 急激に変化 (0.0→0.3 を 1.0→0.0 に変換)
        distance_weight = 1.0 - (normalized_distance / 0.3) ** 0.3  # 急峻な減衰
    else:
        # 中央寄り: ほぼ影響なし
        distance_weight = 0.0
else:  # 通路（坂）
    # 坂: 全範囲(0-1.0)で線形的に変化
    distance_weight = 1.0 - normalized_distance  # 線形減衰

# 方向による重み（境界方向に近いほど影響が大きい）
# cos(θ_b)が1に近い（同じ方向）ほど大きな重み
angle_weight = max(0, (cos(θ_b) + 1) / 2) ** 2  # 0-1の範囲

# 総合的な寄与度
contribution_b = distance_weight * angle_weight
```

### ステップ3: 全境界からの寄与を統合

```python
# すべての境界からの寄与を正規化
total_contribution = sum(contribution_b for all boundaries)

if total_contribution > 0:
    # 重み付き平均で最終高度を計算
    final_height = base_height
    for each boundary b:
        neighbor_height = height[neighbor_group_of_b]
        weight_b = contribution_b / total_contribution
        final_height += (neighbor_height - base_height) * weight_b
else:
    # 影響範囲外（中央部）はベース高度
    final_height = base_height
```

## 期待される効果

1. **滑らかな高度変化**: すべての方向からの寄与を統合することで、鋭利な不連続点が解消
2. **等距離線に沿った正しい高度変化**:
   - 坂: 等距離線ごとに線形的に高度が変化（9→8→7→6→5）
   - 崖: 境界付近の等距離線でのみ急変化、中央は平坦（9→9→9→5→1）
3. **境界からの距離による制御**: 正規化距離(0-1)で崖と坂の特性を表現
4. **方向による滑らかな補間**: 角度重み付けで崖方向と坂方向が滑らかに繋がる

## 実装のポイント

- `contribution_b` がゼロに近い境界は無視（計算効率化）
- 中央部では全境界の影響が小さく、自然とbase_heightに近づく
- 境界付近では最も近い境界の影響が支配的になる
- 複数境界の影響が重なる場合は滑らかに補間される
