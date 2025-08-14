# 遺伝的アルゴリズムでRastrigin関数の最小値を求めるプログラム

# ========= インポート =========
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

# ========= 評価関数 =========
def rastrigin(x: List[float]) -> float:
    A = 10.0
    n = len(x)
    return A * n + sum((xi * xi - A * math.cos(2 * math.pi * xi)) for xi in x)

# ========= 設定 =========
@dataclass
class GAConfig:
    dim: int = 15                       # 次元数
    lower_bound: float = -5.12          # Rastrigin関数の定義域
    upper_bound: float = 5.12           # Rastrigin関数の定義域
    pop_size: int = 500                 # 集団サイズ
    max_generations: int = 1500          # 最大世代数
    elite_ratio: float = 0.02           # エリート選択の割合
    tournament_k: int = 3               # トーナメント選択の候補数
    crossover_rate: float = 0.9         # 交叉率
    mutation_rate: float = 1.0/dim       # 突然変異率
    sbx_eta: float = 20.0               # SBX交叉の指数
    pm_eta: float = 20.0                # 多項式突然変異の指数
    stagnation_patience: int = 240       # 収束判定のための世代数
    random_seed: Optional[int] = 42     # 乱数シード（Noneならランダム）

    # 可視化
    keep_history: bool = True           # 履歴を保持するか
    log_interval: int = 20              # ログ出力の間隔（世代数）
    # リアルタイム
    realtime: bool = True               # リアルタイム可視化を行うか
    refresh_every: int = 5              # 何世代ごとに再描画するか
    show_2d_scatter: bool = True        # dim==2時に散布図を表示する

# ========= 履歴 =========
@dataclass
class GAHistory:
    gens: List[int] = field(default_factory=list)
    best: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    median: List[float] = field(default_factory=list)
    worst: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    best_x: List[List[float]] = field(default_factory=list)

# ========= ユーティリティ =========

# クリップ関数
# xがloとhiの間に収まるようにする
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# 個体の初期化
# random.uniform(a, b)でaからbの間の乱数を生成
# [... for _ in range(cfg.dim)]でcfg.dim次元のリストを生成(リスト内包表記)
def init_individual(cfg: GAConfig) -> List[float]:
    return [random.uniform(cfg.lower_bound, cfg.upper_bound) for _ in range(cfg.dim)]

# 集団の初期化
# init_individual(cfg)をpop_size回呼び出して初期集団を生成
def init_population(cfg: GAConfig) -> List[List[float]]:
    return [init_individual(cfg) for _ in range(cfg.pop_size)]

# 集団の評価
# 処理の流れ
# for ind in pop → 各個体indに対してobjective関数を適用
# objective(ind)で適応度を計算し、リストに格納
def evaluate_population(pop: List[List[float]], objective: Callable[[List[float]], float]) -> List[float]:
    return [objective(ind) for ind in pop]

def tournament_select(pop: List[List[float]], fitness: List[float], k: int) -> List[float]:
    cand_idx = random.sample(range(len(pop)), k)
    best_i = min(cand_idx, key=lambda i: fitness[i])
    return pop[best_i][:]

def sbx_crossover(p1: List[float], p2: List[float], cfg: GAConfig) -> Tuple[List[float], List[float]]:
    c1, c2 = p1[:], p2[:]
    if random.random() > cfg.crossover_rate:
        return c1, c2
    for i in range(cfg.dim):
        if random.random() < 0.5:
            x1, x2 = p1[i], p2[i]
            if x1 == x2:
                c1[i] = x1; c2[i] = x2; continue
            if x1 > x2: x1, x2 = x2, x1
            lb, ub = cfg.lower_bound, cfg.upper_bound
            r = random.random()
            beta = 1.0 + (2.0 * (x1 - lb) / (x2 - x1))
            alpha = 2.0 - beta ** (-(cfg.sbx_eta + 1.0))
            if r <= 1.0/alpha:
                betaq = (r * alpha) ** (1.0/(cfg.sbx_eta + 1.0))
            else:
                betaq = (1.0/(2.0 - r*alpha)) ** (1.0/(cfg.sbx_eta + 1.0))
            child1 = 0.5*((x1+x2) - betaq*(x2-x1))
            child2 = 0.5*((x1+x2) + betaq*(x2-x1))
            c1[i] = clamp(child1, lb, ub)
            c2[i] = clamp(child2, lb, ub)
        else:
            c1[i] = p1[i]; c2[i] = p2[i]
    return c1, c2

# 多項式突然変異
# 個体indの各遺伝子ind[i]に対して、確率mutation_rateで突然変異を適用
# 変異幅は多項式分布に従う
# 変異後の値は、定義域の下限lower_boundと上限upper_boundにクリップされる
def polynomial_mutation(ind: List[float], cfg: GAConfig) -> List[float]:
    lb, ub = cfg.lower_bound, cfg.upper_bound
    for i in range(cfg.dim):
        if random.random() < cfg.mutation_rate:
            x = ind[i]
            if ub - lb <= 0: continue
            delta1 = (x - lb)/(ub - lb)
            delta2 = (ub - x)/(ub - lb)
            r = random.random()
            mut_pow = 1.0/(cfg.pm_eta + 1.0)
            if r < 0.5:
                xy = 1.0 - delta1
                val = 2.0*r + (1.0 - 2.0*r)*(xy ** (cfg.pm_eta + 1.0))
                deltaq = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0*(1.0 - r) + 2.0*(r - 0.5)*(xy ** (cfg.pm_eta + 1.0))
                deltaq = 1.0 - val ** mut_pow
            x = x + deltaq*(ub - lb)
            ind[i] = clamp(x, lb, ub)
    return ind

def _stats(fitness: List[float]) -> Tuple[float,float,float,float,float]:
    arr = np.asarray(fitness, dtype=float)
    return float(np.min(arr)), float(np.mean(arr)), float(np.median(arr)), float(np.max(arr)), float(np.std(arr))

# ========= リアルタイム可視化クラス =========
class RealtimeViz:
    def __init__(self, cfg: GAConfig, objective: Callable[[List[float]], float]):
        self.cfg = cfg
        self.objective = objective
        self.fig = None
        self.ax_curve = None
        self.ax_scatter = None
        self.lines = {}   # "best","mean","median","worst","std" -> Line2D
        self.scatter = None
        self.contour_data = None  # (X, Y, Z)
        self.gens = []
        self.best = []
        self.mean = []
        self.median = []
        self.worst = []
        self.std = []

    def start(self, init_pop: List[List[float]], init_fit: List[float]):
        plt.ion()
        self.fig = plt.figure(figsize=(10,4.5))
        gs = self.fig.add_gridspec(1, 2) if (self.cfg.dim==2 and self.cfg.show_2d_scatter) else self.fig.add_gridspec(1,1)

        # 収束曲線
        self.ax_curve = self.fig.add_subplot(gs[0,0])
        (l_best,)   = self.ax_curve.plot([], [], label="best")
        (l_mean,)   = self.ax_curve.plot([], [], label="mean")
        (l_median,) = self.ax_curve.plot([], [], label="median")
        (l_worst,)  = self.ax_curve.plot([], [], label="worst")
        (l_std,)    = self.ax_curve.plot([], [], linestyle="--", label="std (diversity)")
        self.lines = {"best": l_best, "mean": l_mean, "median": l_median, "worst": l_worst, "std": l_std}
        self.ax_curve.set_xlabel("Generation"); self.ax_curve.set_ylabel("Fitness"); self.ax_curve.set_title("Convergence (Live)")
        self.ax_curve.grid(True, alpha=0.3); self.ax_curve.legend(loc="best")

        # 2D 散布 + 等高線
        if self.cfg.dim == 2 and self.cfg.show_2d_scatter:
            self.ax_scatter = self.fig.add_subplot(gs[0,1])
            lb, ub = self.cfg.lower_bound, self.cfg.upper_bound
            xs = np.linspace(lb, ub, 300)
            ys = np.linspace(lb, ub, 300)
            X, Y = np.meshgrid(xs, ys)
            Z = 20 + (X*X - 10*np.cos(2*np.pi*X)) + (Y*Y - 10*np.cos(2*np.pi*Y))
            self.contour_data = (X, Y, Z)
            self.ax_scatter.contour(X, Y, Z, levels=30, linewidths=0.5)
            p = np.array([(ind[0], ind[1]) for ind in init_pop])
            self.scatter = self.ax_scatter.scatter(p[:,0], p[:,1], s=10, alpha=0.6)
            self.ax_scatter.set_xlim(lb, ub); self.ax_scatter.set_ylim(lb, ub)
            self.ax_scatter.set_xlabel("x0"); self.ax_scatter.set_ylabel("x1")
            self.ax_scatter.set_title("Population on 2D Landscape (Live)")
            self.ax_scatter.grid(True, alpha=0.3)

        # 初期点をプロット
        b, m, med, w, s = _stats(init_fit)
        self._append_stats(0, b, m, med, w, s)
        self._redraw()

    def update(self, gen: int, pop: List[List[float]], fitness: List[float]):
        b, m, med, w, s = _stats(fitness)
        self._append_stats(gen, b, m, med, w, s)
        # 散布更新
        if self.ax_scatter is not None:
            p = np.array([(ind[0], ind[1]) for ind in pop])
            self.scatter.set_offsets(p)  # 現世代の位置に更新
        # 定期的に再描画
        if gen % self.cfg.refresh_every == 0 or gen == 1:
            self._redraw()

    def finish(self):
        self._redraw()
        plt.ioff()
        plt.show()

    def _append_stats(self, gen, b, m, med, w, s):
        self.gens.append(gen)
        self.best.append(b); self.mean.append(m); self.median.append(med); self.worst.append(w); self.std.append(s)

    def _redraw(self):
        # 収束曲線のデータ更新
        x = np.array(self.gens)
        self.lines["best"].set_data(x, self.best)
        self.lines["mean"].set_data(x, self.mean)
        self.lines["median"].set_data(x, self.median)
        self.lines["worst"].set_data(x, self.worst)
        self.lines["std"].set_data(x, self.std)
        # 軸範囲を自動調整（多少の余白）
        if len(x) > 1:
            self.ax_curve.set_xlim(0, max(x))
            yall = np.concatenate([np.array(self.best), np.array(self.mean), np.array(self.median), np.array(self.worst)])
            ymin, ymax = float(np.min(yall)), float(np.max(yall))
            if ymin == ymax: ymax = ymin + 1.0
            pad = 0.05*(ymax - ymin)
            self.ax_curve.set_ylim(ymin - pad, ymax + pad)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

# ========= 進化ループ（可視化対応） =========
def run_ga(objective: Callable[[List[float]], float],
           cfg: GAConfig) -> Tuple[List[float], float, int, Optional[GAHistory]]:
    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)

    history = GAHistory() if cfg.keep_history else None
    viz = RealtimeViz(cfg, objective) if cfg.realtime else None

    pop = init_population(cfg)
    fitness = evaluate_population(pop, objective)

    # リアルタイム初期化
    if viz is not None:
        viz.start(pop, fitness)

    elite_n = max(1, int(math.ceil(cfg.pop_size * cfg.elite_ratio)))
    best_idx = min(range(len(pop)), key=lambda i: fitness[i])
    best_ind = pop[best_idx][:]
    best_fit = fitness[best_idx]
    best_gen = 0
    no_improve = 0

    # 履歴0世代
    if history:
        b, m, med, w, s = _stats(fitness)
        history.gens.append(0)
        history.best.append(b); history.mean.append(m); history.median.append(med)
        history.worst.append(w); history.std.append(s)
        history.best_x.append(best_ind[:])

    for gen in range(1, cfg.max_generations + 1):
        # エリート
        elites_idx = sorted(range(len(pop)), key=lambda i: fitness[i])[:elite_n]
        elites = [pop[i][:] for i in elites_idx]

        # 新集団
        new_pop: List[List[float]] = []
        new_pop.extend(elites)
        while len(new_pop) < cfg.pop_size:
            p1 = tournament_select(pop, fitness, cfg.tournament_k)
            p2 = tournament_select(pop, fitness, cfg.tournament_k)
            c1, c2 = sbx_crossover(p1, p2, cfg)
            c1 = polynomial_mutation(c1, cfg); 
            if len(new_pop) < cfg.pop_size: new_pop.append(c1)
            c2 = polynomial_mutation(c2, cfg);
            if len(new_pop) < cfg.pop_size: new_pop.append(c2)

        pop = new_pop
        fitness = evaluate_population(pop, objective)

        # ベスト更新判定
        curr_best_idx = min(range(len(pop)), key=lambda i: fitness[i])
        curr_best_fit = fitness[curr_best_idx]
        if curr_best_fit + 1e-12 < best_fit:
            best_fit = curr_best_fit
            best_ind = pop[curr_best_idx][:]
            best_gen = gen
            no_improve = 0
        else:
            no_improve += 1

        # 履歴
        if history:
            b, m, med, w, s = _stats(fitness)
            history.gens.append(gen)
            history.best.append(b); history.mean.append(m); history.median.append(med)
            history.worst.append(w); history.std.append(s)
            history.best_x.append(best_ind[:])

        # リアルタイム更新
        if viz is not None:
            viz.update(gen, pop, fitness)

        # ログ
        if gen % cfg.log_interval == 0 or gen == 1:
            print(f"[Gen {gen:3d}] best={best_fit:.6f}  mean={np.mean(fitness):.6f}  std={np.std(fitness):.6f}  last_improved@{best_gen}")

        # 早期終了
        if no_improve >= cfg.stagnation_patience:
            print(f"Stop early: no improvement for {cfg.stagnation_patience} generations.")
            break

    if viz is not None:
        viz.finish()

    return best_ind, best_fit, best_gen, history

# ========= 実行 =========
if __name__ == "__main__":
    cfg = GAConfig(
        dim=2,                   # 2にすると右側の散布が出ます。>=3でも収束曲線はライブ更新されます
        lower_bound=-5.12,
        upper_bound=5.12,
        pop_size=120,
        max_generations=300,
        elite_ratio=0.02,
        tournament_k=3,
        crossover_rate=0.9,
        mutation_rate=1.0/10,
        sbx_eta=20.0,
        pm_eta=20.0,
        stagnation_patience=80,
        random_seed=42,
        keep_history=True,
        log_interval=20,
        realtime=True,          # ← ライブ描画ON
        refresh_every=5,        # ← 5世代ごとに再描画
        show_2d_scatter=True,   # ← dim==2のとき散布図表示
    )

    best_x, best_f, best_gen, history = run_ga(rastrigin, cfg)
    print("\n=== RESULT ===")
    print(f"best fitness  : {best_f:.12f}")
    print(f"found at gen  : {best_gen}")
    print(f"best solution : {['{:.12f}'.format(v) for v in best_x]}")
