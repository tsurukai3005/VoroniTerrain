"""
改善版v3の坂・崖生成をテストするスクリプト

改善内容(v3):
- 処理順序の変更: 坂を先に計算 → その後で崖を適用
- 崖の上下判定: 坂で傾斜した後の実際の高度で判定
- 高度逆転の解消: 同程度の高度の領域間で不自然な崖が立ち上がる問題を修正

改善内容(v2):
- 境界を上り/下り/崖に分類
- 上り境界から下り境界への滑らかな遷移
- 領域中心での急峻な立ち上がりを解消
"""
import sys
import time

print("=" * 60)
print("改善版地形生成テスト")
print("=" * 60)

start_time = time.time()

# メインスクリプトをインポート
from voronoi_terrain import VoronoiTerrain, visualize_2d, visualize_3d, visualize_slope_angles, print_statistics

print("\n地形を生成中...")
terrain = VoronoiTerrain()

elapsed = time.time() - start_time
print(f"\n生成完了 (所要時間: {elapsed:.2f}秒)")
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
print("テスト完了!")
print("=" * 60)
print("\n改善点の確認:")
print("1. terrain_2d.png - 坂の方向と境界の種別を確認")
print("2. terrain_slope_angles.png - 傾斜角度の分布を確認")
print("3. terrain_3d.html - 3D表示で滑らかさを確認")
print("\n期待される結果(v3):")
print("✓ 領域中心の急峻な立ち上がりが解消")
print("✓ 上り境界→下り境界への滑らかな遷移")
print("✓ 崖は実際の高度で上下判定(不自然な立ち上がり解消)")
print("✓ 同程度の高度の領域間で自然な境界")
print("✓ 坂→崖の正しい処理順序")
print("=" * 60)
