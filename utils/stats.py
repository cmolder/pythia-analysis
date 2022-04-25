import numpy as np
import pandas as pd
from utils import utils
from itertools import product
from IPython.display import display # DEBUG

def get_longest_simpoints(weights):
    idx = (weights.groupby('trace')['weight'].transform(max) == weights['weight'])
    traces = weights[idx].trace
    return traces

def _process_prefetcher(stats, df, weights, tr, l1d_pref, l2c_pref, llc_pref, pyt_level_th):
    wt = weights[weights.trace == tr][['simpoint', 'weight']]
    data = df[(df.trace == tr) & (df.l1d_pref == l1d_pref) & (df.l2c_pref == l2c_pref) & (df.llc_pref == llc_pref)].copy()
    data.pythia_level_threshold.fillna(float('-inf'), inplace=True)
    if pd.isna(pyt_level_th):
        pyt_level_th = float('-inf')
    data = data[(data.pythia_level_threshold == pyt_level_th)]
    data = data.merge(wt, on='simpoint')
    weights = data['weight'] / sum(data['weight'])

    stats['trace'] = np.append(stats['trace'], tr)
    stats['l1d_pref'] = np.append(stats['l1d_pref'], l1d_pref)
    stats['l2c_pref'] = np.append(stats['l2c_pref'], l2c_pref)
    stats['llc_pref'] = np.append(stats['llc_pref'], llc_pref)
    stats['pythia_level_threshold'] = np.append(stats['pythia_level_threshold'], pyt_level_th)
    stats['simpoint'] = np.append(stats['simpoint'], 'weighted')

    if len(data) == 0:
        #print(f'[DEBUG] {pf} {tr} not found')
        for metric in utils.metrics:
            stats[f'{metric}'] = np.append(stats[f'{metric}'], np.nan)
        return
    
    for metric in utils.metrics:
        target = data[metric].item() if len(data) <= 1 else utils.mean(data[metric], metric, weights=weights)
        stats[f'{metric}'] = np.append(stats[f'{metric}'], target)

    
def get_weighted_statistics(df, weights):
    stats = {
        'trace': np.array([]),
        'l1d_pref': np.array([]),
        'l2c_pref': np.array([]),
        'llc_pref': np.array([]),
        'pythia_level_threshold': np.array([]),
        'simpoint': np.array([]),
        'accuracy': np.array([]),
        'coverage': np.array([]),
        'ipc_improvement': np.array([]),
        'mpki_reduction': np.array([]),
        'dram_bw_reduction': np.array([])
    }
    

    for tr in df.trace.unique():
        for l1p, l2p, llp, plt in product(df.l1d_pref.unique(), df.l2c_pref.unique(), df.llc_pref.unique(), df.pythia_level_threshold.unique()):
            _process_prefetcher(stats, df, weights, tr, l1p, l2p, llp, plt)
               
    return pd.DataFrame(stats)