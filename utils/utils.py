import pandas as pd
import numpy as np
from scipy import stats

metrics = ['L2C_accuracy', 'L2C_coverage', 'LLC_accuracy', 'LLC_coverage', 'ipc_improvement', 'L2C_mpki_reduction', 'LLC_mpki_reduction', 'dram_bw_reduction']
amean_metrics = ['L2C_accuracy', 'L2C_coverage', 'LLC_accuracy', 'LLC_coverage', 'dram_bw_reduction']

gap = [
    'cc', 'pr', 'sssp', 'bfs', 'tc'
]
spec = [
    'astar', 'bwaves', 'bzip2', 'cactusADM', 'calculix',
    'gcc', 'GemsFDTD', 'hmmer', 'lbm', 'leslie3d',
    'libquantum', 'mcf', 'milc', 'omnetpp', 'soplex',
    'sphinx3', 'tonto', 'wrf', 'xalancbmk'
]
cloudsuite = [
    'cassandra_core0', 'cassandra_core1', 'cassandra_core2', 'cassandra_core3',
    'classification_core0', 'classification_core1', 'classification_core2', 'classification_core3',
    'cloud9_core0', 'cloud9_core1', 'cloud9_core2', 'cloud9_core3',
    'nutch_core0', 'nutch_core1', 'nutch_core2', 'nutch_core3',
    'streaming_core0', 'streaming_core1', 'streaming_core2', 'streaming_core3'
]

def read_weights_file(path):
    weights = pd.read_csv(path, sep=' ', header=None)
    weights.columns = ['full_trace', 'weight']
    
    trace = []
    simpoint = []
    
    for tr in weights.full_trace:
        tokens = tr.split('_')
        
        if len(tokens) == 3: # Cloudsuite
            trace.append(tokens[0] + '_' + tokens[2])
            simpoint.append(tokens[1])
        if len(tokens) == 2: # SPEC '06
            trace.append(tokens[0])
            simpoint.append(tokens[1])
        if len(tokens) == 1: # Gap
            trace.append(tokens[0])
            simpoint.append('default')
    
    weights['trace'] = trace
    weights['simpoint'] = simpoint
    
    return weights


def read_degree_sweep_file(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace('scooby_double', 'pythia_double')
    df.columns = df.columns.str.replace('scooby', 'pythia')
    df.columns = df.columns.str.replace('spp_dev2', 'spp')
    df.columns = df.columns.str.replace('bop', 'bo')
    return df


def read_data_file(path):
    df = pd.read_csv(path)
    df.simpoint = df.simpoint.fillna('default')
    
    for col in ['L1D_pref', 'L2C_pref', 'LLC_pref']:
    
        df[col] = df[col].replace({
            'scooby_double': 'pythia_double',
            'scooby': 'pythia',
            'spp_dev2': 'spp',
            'bop': 'bo'
        }, regex=True)

        # Fix prefetcher ordering
        df[col] = df[col].apply(lambda c : '_'.join(sorted(c.split('_'))))
    
    
    df['all_pref'] = list(zip(df.L1D_pref, df.L2C_pref, df.LLC_pref))
    return df

def mean(values, metric, weights=None):
    if type(weights) is np.ndarray:
        assert np.isclose(np.sum(weights), 1), 'Weights should sum to 1'
    if metric in amean_metrics:
        return np.average(values, weights=weights)
    else:
        if 'ipc_improvement' in metric:
            # Add 100 to prevent negative values (so that 100 = no prefetcher baseline)
            return stats.gmean(values + 100, weights=weights) - 100 
        if 'mpki_reduction' in metric:
            # Take gmean of relative misses instead of MPKI reduction to prevent negative values
            return 100 - stats.gmean(100 - values, weights=weights) 
        print(f'Unknown metric {metric}')
        
def rank_prefetchers(df, metric, count=None):
    """Return the <count> best prefetchers, in order of maximum <metric>.
    """
    pf_avgs = []
    # Filter out opportunity prefetchers from the ranking.
    
    for col in ['L1D_pref', 'L2C_pref', 'LLC_pref']:
        df = df[~df[col].str.startswith('pc_comb') & ~df[col].str.contains('phase_comb')]
    
    for i, (pf, df_pf) in enumerate(df.groupby(['L1D_pref', 'L2C_pref', 'LLC_pref'])):
        avg = mean(df_pf[metric], metric)
        pf_avgs.append((avg, pf))
        
    best = sorted(pf_avgs)[::-1]
    if count != None:
        best = best[:count]
    
    return [pf for _, pf in best]