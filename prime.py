import numpy as np
import zlib
import pandas as pd
from sympy import primerange

# ===== Helper functions =====
def gen_prime(n):
    return list(primerange(0, n))[:n]

def gen_random(n, seed=42):
    np.random.seed(seed)
    return np.random.randint(1, 1e6, size=n).tolist()

def log_estimate(index):
    # Using i*log(i) as a simple structural model
    return [int(i * np.log(i)) if i > 1 else 0 for i in range(1, len(index) + 1)]

def residual(seq, model):
    return [x - y for x, y in zip(seq, model)]

def compress_bytes(data):
    return len(zlib.compress(data))

def to_bytes(seq):
    # Map integer to bytes (4-byte little endian signed)
    return b''.join(int(x).to_bytes(4, byteorder='little', signed=True) for x in seq)

def compute_lambdak(x):
    model = log_estimate(range(1, len(x) + 1))
    delta = residual(x, model)
    Cx = compress_bytes(to_bytes(x))
    Cd = compress_bytes(to_bytes(delta))
    return round(Cd / Cx, 4), Cx, Cd

# ===== Experiment Runner =====
def run_experiment(n_list):
    records = []
    for n in n_list:
        primes = gen_prime(n)
        rnd = gen_random(n)

        lambda_prime, cx_p, cd_p = compute_lambdak(primes)
        lambda_rand, cx_r, cd_r = compute_lambdak(rnd)

        records.append({
            'n': n,
            'lambda_k_prime': lambda_prime,
            'lambda_k_random': lambda_rand,
            'Cx_prime': cx_p,
            'Cd_prime': cd_p,
            'Cx_random': cx_r,
            'Cd_random': cd_r
        })

    df = pd.DataFrame(records).set_index('n')
    return df

# ===== Run & Display =====
n_values = [100, 500, 1000, 2000, 5000, 10000, 50000, 100000]
df_result = run_experiment(n_values)
df_result
