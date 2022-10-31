from itertools import product

from utils import utils

def add_means(df):
    """Add mean statistics across a series of benchmarks.

    For weighted SimPoints, see <script.py> eval -w in the Pythia
    repository."""
    # Don't apply mean more than once.
    if 'mean' in df.run_name.values:
        return df

    df = df.copy()

    simpoint_cols = [f'cpu{cpu}_simpoint' for cpu in range(max(df.num_cpus))]
    full_trace_cols = [f'cpu{cpu}_full_trace' for cpu in range(max(df.num_cpus))]
    trace_cols = [f'cpu{cpu}_trace' for cpu in range(max(df.num_cpus))]

    caches = [
        'L1D', 'L2C', 'LLC',
        *(f'cpu{cpu}_L1D' for cpu in range(max(df.num_cpus))),
        *(f'cpu{cpu}_L2C' for cpu in range(max(df.num_cpus))),
    ]

    for pf, seed in product(df.all_pref.unique(), df.seed.unique()):
        df_ = df[(df.all_pref == pf) & (df.seed == seed)]
        row = df_.iloc[-1].copy()

        for col in ['run_name', *simpoint_cols, *trace_cols, *full_trace_cols]:
            row[col] = 'mean'

        for metric in [
            'ipc',
            'ipc_improvement',
            *(f'cpu{cpu}_ipc' for cpu in range(max(df.num_cpus))),
            *(f'cpu{cpu}_ipc_improvement' for cpu in range(max(df.num_cpus))),
            *(f'{cache}_accuracy' for cache in caches),
            *(f'{cache}_coverage' for cache in caches),
            *(f'{cache}_mpki_reduction' for cache in caches),
            *(f'{cache}_issued_prefetches' for cache in caches)]:
            row[metric] = utils.mean(df_[metric], metric)
        df = df.append(row)
    return df
