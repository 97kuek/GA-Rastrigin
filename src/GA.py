# ==========================================================
# 遺伝的アルゴリズム(GA)によるRastrigin関数の最小値探索
# ==========================================================

import math
import random
from dataclasses import dataclass
from typing import List, Callable, Optional, Dict, Any
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

# ------------------ 評価関数（目的関数） ------------------
def rastrigin(x: List[float]) -> float:
    A = 10.0
    n = len(x)
    return A * n + sum((xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x)

# ------------------ GA（遺伝的アルゴリズム）の設定 ------------------
@dataclass
class GAConfig:
    # 探索空間など
    dim: int = 2                        # 変数の次元（ここでは2次元）
    lower_bound: float = -5.12          # 探索範囲の下限
    upper_bound: float =  5.12          # 探索範囲の上限

    # 初期集団
    init_low: float = -5.12
    init_high: float = -2.0

    # GAの規模
    pop_size: int = 30                  # 集団サイズ
    max_generations: int = 100          # 最大世代数
    offspring_per_gen: int = 60         # 1世代で生成する子の数（交叉→突然変異）

    # 交叉パラメータ
    blx_alpha: float = 0.5              # BLX-α の α（親の区間 ±αI まで一様サンプル）

    # 突然変異パラメータ
    mutation_prob: float = 0.2          # 突然変異の確率（各遺伝子に対して）
    mutation_sigma_ratio: float = 0.08  # 範囲幅 * 0.08 を標準偏差に

    # その他
    random_seed: Optional[int] = 42     # 再現性のため固定（Noneにすると毎回ランダム）
    log_interval: int = 10              # ログ表示間隔（世代単位）

# ------------------ ユーティリティ関数群 ------------------
def clamp(x: float, lo: float, hi: float) -> float:
    """探索範囲を超えた場合に、範囲の端に制限（致死個体を防ぐ有効解修復）"""
    return max(lo, min(hi, x))

def init_individual(cfg: GAConfig) -> List[float]:
    """初期個体をランダム生成（init_low〜init_high の一様分布）"""
    return [random.uniform(cfg.init_low, cfg.init_high) for _ in range(cfg.dim)]

def init_population(cfg: GAConfig) -> List[List[float]]:
    """初期集団（pop_size 個の個体）を生成"""
    return [init_individual(cfg) for _ in range(cfg.pop_size)]

def evaluate_population(pop: List[List[float]], objective: Callable[[List[float]], float]) -> List[float]:
    """全個体の目的関数値（小さいほど良い）を計算"""
    return [objective(ind) for ind in pop]

def roulette_select(pop: List[List[float]], fitness: List[float]) -> List[float]:
    """
    ルーレット選択（最小化版）
    - 値が小さい個体ほど選ばれやすいように重みを付ける
    - 重み s_i = f_max - f_i として、非負＆小さい f_i ほど大きい s_i になる
    """
    fmax = max(fitness)                                                                     # 最大適応度（最小化なので最小値を求める）
    scores = [fmax - f for f in fitness]                                                    # スコアを計算（小さいほど選ばれやすい）
    ssum = sum(scores)                                                                      # スコアの合計
    if ssum <= 0:
        # もし全員同じスコアなら一様ランダム
        return pop[random.randrange(len(pop))][:]
    r = random.random() * ssum
    acc = 0.0
    for ind, s in zip(pop, scores): # 個体とスコアを同時にループ
        acc += s
        if acc >= r:
            return ind[:]
    return pop[-1][:]  # 念のためのフォールバック

def blx_alpha_crossover(p1: List[float], p2: List[float], cfg: GAConfig) -> List[float]:
    """
    BLX-α 交叉
    - 親2個体の i番目の遺伝子 x1, x2 から、
      [min-αI, max+αI]（I=|x2-x1|）の一様分布で子 c_i をサンプル
    - 端に出たら clamp で修復
    """
    alpha = cfg.blx_alpha
    lb, ub = cfg.lower_bound, cfg.upper_bound
    child: List[float] = []
    for i in range(cfg.dim):
        x1, x2 = p1[i], p2[i]
        cmin, cmax = (x1, x2) if x1 <= x2 else (x2, x1)
        I = cmax - cmin
        low  = cmin - alpha * I
        high = cmax + alpha * I
        val = random.uniform(low, high)
        child.append(clamp(val, lb, ub))
    return child

def gaussian_mutation(ind: List[float], cfg: GAConfig) -> List[float]:
    """
    ガウス変異
    - 個体の各遺伝子について確率 mutation_prob の確率で突然変異を起こす
    - 突然変異が起こる場合、N(0, sigma^2) に従う乱数を現在の遺伝子の値に加算する
    - sigma は探索範囲幅に対する比（mutation_sigma_ratio）から算出
    - 範囲外に出たら clamp で修復
    """
    lb, ub = cfg.lower_bound, cfg.upper_bound                   # 探索範囲の下限と上限
    width = ub - lb                                             # 探索範囲の幅
    sigma = cfg.mutation_sigma_ratio * width                    # 例：幅×0.08
    new_ind = ind[:]                                            # 元の個体をコピーして変異後の個体を作成
    for i in range(cfg.dim):
        if random.random() < cfg.mutation_prob:                 # 突然変異の確率が満たされた場合
            mutated = new_ind[i] + random.gauss(0.0, sigma)     # N(0, sigma^2) の乱数を加える
            new_ind[i] = clamp(mutated, lb, ub)                 # 範囲外に出たら clamp で修復
    return new_ind

# ------------------ GA本体（MGGモデル） ------------------
def run_ga_mgg_with_snapshots(objective: Callable[[List[float]], float], cfg: GAConfig):
    """
    MGG（Minimal Generation Gap）で世代を少しずつ更新する GA
    1. 親選択：エリート1 + ルーレット1
    2. 子生成：BLX-α で offspring_per_gen 体 → ガウス変異で微調整
    3. 生存選択：親2 + 子 の中から最良2体を選ぶ
    4. 置換：集団内の2枠だけを置き換える（= ギャップ最小）
    """

    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)

    # 初期集団と評価
    pop = init_population(cfg)
    fitness = evaluate_population(pop, objective)

    # ベストの初期化
    best_idx = min(range(len(pop)), key=lambda i: fitness[i])
    best_ind = pop[best_idx][:]
    best_fit = fitness[best_idx]
    best_gen = 0

    # 可視化用スナップショット
    snaps: List[Dict[str, Any]] = []

    # === 世代ループ ===
    for gen in range(1, cfg.max_generations + 1):
        # ---- 親選択 ----
        elite_i = min(range(len(pop)), key=lambda i: fitness[i])  # 集団内ベスト
        elite = pop[elite_i][:]
        roulette = roulette_select(pop, fitness)                  # ルーレットでもう1親

        # ---- 子個体生成（交叉 → 突然変異）----
        offspring: List[List[float]] = []
        for _ in range(cfg.offspring_per_gen):
            child = blx_alpha_crossover(elite, roulette, cfg)  # BLX-αで親の区間±αIからサンプル
            child = gaussian_mutation(child, cfg)              # ★微小なランダム揺らぎで探索の幅を確保
            offspring.append(child)

        # ---- 生存選択（親2 + 子 → 最良2体）----
        candidates = offspring + [elite, roulette]
        cand_fit = [objective(ind) for ind in candidates]
        order = sorted(range(len(candidates)), key=lambda i: cand_fit[i])[:2]
        surv1 = candidates[order[0]][:]  # 最良
        surv2 = candidates[order[1]][:]  # 次点

        # ---- 集団の一部を置き換える（MGGのミニマル更新）----
        pop[elite_i] = surv1
        roulette_j = (elite_i + 1) % cfg.pop_size  # シンプルに隣のスロットへ
        pop[roulette_j] = surv2

        # ---- 評価更新 ----
        fitness = evaluate_population(pop, objective)

        # ---- グローバルベスト更新 ----
        curr_best_i = min(range(len(pop)), key=lambda i: fitness[i])
        if fitness[curr_best_i] < best_fit:
            best_fit = fitness[curr_best_i]
            best_ind = pop[curr_best_i][:]
            best_gen = gen

        # ---- 可視化用スナップショット保存（座標のみ）----
        snaps.append({
            "gen": gen,
            "pop": np.array(pop, dtype=float),
            "best": np.array(best_ind, dtype=float),
            "parents": np.array([elite, roulette], dtype=float),
            "offspring": np.array(offspring, dtype=float),  # 交叉→突然変異後の子
        })

        # ---- ログ ----
        if gen % cfg.log_interval == 0 or gen == 1:
            print(f"[Gen {gen:3d}] best={best_fit:.6f}  found@{best_gen}")

    return best_ind, best_fit, best_gen, snaps

# ------------------ アニメーション生成 ------------------
def make_animation(snaps: List[Dict[str, Any]], cfg: GAConfig, path: Path):
    """
    世代ごとに保存したスナップショットを用いて、個体分布の推移をGIF化。
    - 青：集団
    - 赤：最良個体
    - オレンジ/紫：親
    - 灰：子（交叉→突然変異後）
    """
    lb, ub = cfg.lower_bound, cfg.upper_bound
    fig, ax = plt.subplots(figsize=(6, 6))                                                          # グラフのサイズを設定
    ax.set_xlim(lb, ub)                                                                             # x軸の範囲を設定
    ax.set_ylim(lb, ub)                                                                             # y軸の範囲を設定
    ax.set_xlabel("x")                                                                              # x軸ラベル
    ax.set_ylabel("y")                                                                              # y軸ラベル
    ax.set_title("Genetic Algorithm (BLX-α + Gaussian Mutation, MGG)")                              # タイトル設定

    # 各描画レイヤ
    scat_pop = ax.scatter([], [], s=25, color="blue", label="Population")
    scat_best = ax.scatter([], [], s=90, marker="*", color="red", label="Best Individual")
    scat_p1 = ax.scatter([], [], s=60, marker="s", color="orange", label="Parent 1 (Elite)")
    scat_p2 = ax.scatter([], [], s=60, marker="D", color="purple", label="Parent 2 (Roulette)")
    scat_off = ax.scatter([], [], s=8, alpha=0.35, color="gray", label="Offspring")
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

    ax.legend(loc="upper right")

    # 初期フレーム（空）
    def init():
        empty = np.empty((0, 2))
        scat_pop.set_offsets(empty)
        scat_best.set_offsets(empty)
        scat_p1.set_offsets(empty)
        scat_p2.set_offsets(empty)
        scat_off.set_offsets(empty)
        txt.set_text("")
        return scat_pop, scat_best, scat_p1, scat_p2, scat_off, txt

    # 各フレームの更新
    def update(frame_idx: int):
        snap = snaps[frame_idx]
        scat_pop.set_offsets(snap["pop"])
        scat_best.set_offsets(snap["best"].reshape(1, 2))
        scat_p1.set_offsets(snap["parents"][0].reshape(1, 2))
        scat_p2.set_offsets(snap["parents"][1].reshape(1, 2))
        scat_off.set_offsets(snap["offspring"])  # 交叉→突然変異後の子
        txt.set_text(f"Gen {snap['gen']}")
        return scat_pop, scat_best, scat_p1, scat_p2, scat_off, txt

    # 少しゆっくり目の速度（必要に応じて interval / fps を調整）
    anim = FuncAnimation(fig, update, frames=len(snaps), init_func=init, blit=False, interval=200)
    writer = PillowWriter(fps=5)

    # 保存ディレクトリが無ければ作成（assets は自動生成）
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(path), writer=writer)
    plt.close(fig)
    return path

# ------------------ 実行（エントリポイント） ------------------
if __name__ == "__main__":
    # GAの設定を適用
    cfg = GAConfig()

    # GIF保存先パス
    gif_path = Path(__file__).parent.parent / "assets" / "GA_result.gif"

    # GA 実行
    best_x, best_f, best_gen, snaps = run_ga_mgg_with_snapshots(rastrigin, cfg)

    # GIF 生成
    gif_file = make_animation(snaps, cfg, gif_path)

    # 結果表示
    print("\n=== 結果 ===")
    print(f"最良適応度 : {best_f:.6f}")
    print(f"発見世代   : {best_gen}")
    print(f"最良解     : {best_x}")
    print(f"GIF保存先  : {gif_file}")
