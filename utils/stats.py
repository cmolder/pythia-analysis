import numpy as np
import pandas as pd
from utils import utils
from itertools import product
from IPython.display import display # DEBUG

def get_longest_simpoints(weights):
    idx = (weights.groupby('trace')['weight'].transform(max) == weights['weight'])
    traces = weights[idx].trace
    return traces

def _process_prefetcher(stats, df, weights, tr, pf, plt):
    wt = weights[weights.trace == tr][['simpoint', 'weight']]
    data = df[(df.trace == tr) & (df.all_pref == pf) & (df.pythia_level_threshold == plt)]
    data = data.merge(wt, on='simpoint')
    weights = data['weight'] / sum(data['weight'])

    stats['trace'] = np.append(stats['trace'], tr)
    stats['all_pref'].append(pf)
    stats['simpoint'] = np.append(stats['simpoint'], 'weighted')
    stats['pythia_level_threshold'] = np.append(stats['pythia_level_threshold'], plt)
    
    if len(data) == 0:
        print(f'[DEBUG] {pf} {tr} {plt} not found')
        for metric in utils.metrics:
            stats[f'{metric}'] = np.append(stats[f'{metric}'], np.nan)
        return
    
    for metric in utils.metrics:
        target = data[metric].item() if len(data) <= 1 else utils.mean(data[metric], metric, weights=weights)
        stats[f'{metric}'] = np.append(stats[f'{metric}'], target)
        #print('[DEBUG]', pf, metric, data[metric].to_list(), weights.to_list(), stats[f'{metric}'][-1])

    
def get_weighted_statistics(df, weights):
    stats = {
        'trace': np.array([]),
        'all_pref': [],
        'pythia_level_threshold': np.array([]),
        'simpoint': np.array([]),
        'L2C_accuracy': np.array([]),
        'L2C_coverage': np.array([]),
        'LLC_accuracy': np.array([]),
        'LLC_coverage': np.array([]),
        'ipc_improvement': np.array([]),
        'L2C_mpki_reduction': np.array([]),
        'LLC_mpki_reduction': np.array([]),
        'dram_bw_reduction': np.array([])
    }
    
    df.pythia_level_threshold = df.pythia_level_threshold.fillna('None')

    for tr in df.trace.unique():
        for pf, plt in product(df.all_pref.unique(), df.pythia_level_threshold.unique()):
            _process_prefetcher(stats, df, weights, tr, pf, plt)
           
    stats = pd.DataFrame(stats)
    stats.pythia_level_threshold.replace('None', float('-inf'), inplace=True)
    stats.pythia_level_threshold = stats.pythia_level_threshold.astype(float)
    return stats