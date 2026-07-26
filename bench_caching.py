"""Benchmark 5 configs to measure real per-config time with caching."""
import time
import cupy as cp
import numpy as np
from . import config as cfg
from .initial_conditions_mv import compute_path_ordered_fund_Wilson_line
from .evolution_single import jimwlk_evolution

cfg.N = 512
cfg.a = cfg.l / cfg.N
cfg.a2 = cfg.a**2
cfg.variance_of_mv_noise = float(np.sqrt(cfg.mu2 / (cfg.Ny * cfg.a2)))
cfg.Y_f = 5.0
cfg.dY = 0.01

# warmup (triggers caching of D and K_of_k)
rng = cp.random.Generator(cp.random.XORWOW(seed=0))
V = compute_path_ordered_fund_Wilson_line(rng)
_ = jimwlk_evolution(V, rng=rng, measure_interval=999)
cp.cuda.Stream.null.synchronize()
print("warmup done\n")

for i in range(5):
    rng = cp.random.Generator(cp.random.XORWOW(seed=100 + i))
    t0 = time.time()
    V = compute_path_ordered_fund_Wilson_line(rng)
    cp.cuda.Stream.null.synchronize()
    t_init = time.time() - t0
    t1 = time.time()
    result = jimwlk_evolution(V, rng=rng, measure_interval=999)
    cp.cuda.Stream.null.synchronize()
    t_evol = time.time() - t1
    t_total = t_init + t_evol
    print("config %d: init=%.3fs  evol=%.3fs  total=%.3fs" % (i, t_init, t_evol, t_total))
