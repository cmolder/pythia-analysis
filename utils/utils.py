from typing import Optional

import pandas as pd
import numpy as np
from scipy import stats

# Selecting suites
suites = {
    'irregular': ['astar', 'bfs', 'cc', 'mcf', 'omnetpp', 'pr', 'soplex', 
                  'sphinx3', 'xalancbmk'],
    'spec06': ['astar', 'bwaves', 'cactusADM', 'GemsFDTD', 'lbm', 'leslie3d', 
               'libquantum', 'mcf', 'milc', 'omnetpp', 'soplex', 'sphinx3', 
               'xalancbmk', 'zeusmp'],
               # The below benchmarks have <= 3 LLC MPKI on the baseline.
               #'bzip2', 'calculix', 'gcc', 'gobmk', 'gromacs', 'h264ref', 
               #'hmmer', 'perlbench', 'tonto', 'wrf'],
    'gap': ['bc', 'bfs', 'cc', 'pr', 'sssp', 'tc'],
    'cloudsuite': ['cassandra', 'classifcation', 'cloud9', 'nutch', 'streaming']
}

# Selecting phases
phases = {}
phases['one_phase'] = {
    # SPEC 06
    'astar': '313B',
    'bwaves': '1861B',
    'bzip2': '183B',
    'cactusADM': '734B',
    'calculix': '2670B',
    'gcc': '13B',
    'GemsFDTD': '109B',
    'gobmk': '135B',
    'gromacs': '1B',
    'h264ref': '273B',
    'hmmer': '7B',
    'lbm': '94B',
    'leslie3d': '1116B',
    'libquantum': '1210B',
    'mcf': '46B',
    'milc': '360B',
    'omnetpp': '340B',
    'perlbench': '53B',
    'soplex': '66B',
    'sphinx3': '2520B',
    'tonto': '2834B',
    'wrf': '1212B',
    'xalancbmk': '99B',
    'zeusmp': '600B',
    # GAP
    'bc': 'default',
    'bfs': 'default',
    'cc': 'default',
    'pr': 'default',
    'sssp': 'default',
    'tc': 'default',
    # Cloudsuite
    'cassandra': 'phase0',
    'classifcation': 'phase0',
    'cloud9': 'phase0',
    'nutch': 'phase0',
    'streaming': 'phase0'
}
phases['weighted'] = {k: 'weighted' for k in phases['one_phase']}


def read_data_file(path: str):
    """Read a stats csv generated by the evaluate script in Pythia/experiments.

    Parameters:
        path: Path to the csv file.

    Returns:
        df: A minimally-processed dataframe of the stats csv.
    """
    df = pd.read_csv(path)

    # Fill nan values
    simpoint_cols = [f'cpu{cpu}_simpoint' for cpu in range(max(df.num_cpus))]
    df[simpoint_cols] = df[simpoint_cols].fillna('default')

    # Clean prefetcher names, fix prefetcher ordering
    for col in ['L1D_pref', 'L2C_pref', 'LLC_pref']:
        df[col] = df[col].replace({
            'cygnus_state': 'cygnusstate',
            'from_file': 'fromfile',
            'ip_stride': 'ipstride',
            'isb_ideal': 'isbideal',
            'isb_real': 'isbreal',
            'next_line': 'nextline',
            'spp_dev2': 'sppdev2'
        }, regex=True)

        # Fix prefetcher ordering
        df[col] = df[col].apply(lambda c: '_'.join(sorted(c.split('_'))))

        # Unfix some orderings
        df[col] = df[col].replace({
            'cygnusstate': 'cygnus_state',
            'fromfile': 'from_file',
            'ipstride': 'ip_stride',
            'isbideal': 'isb_ideal',
            'isbreal': 'isb_real',
            'nextline': 'next_line',
            'sppdev2': 'spp_dev2'
        })

    # Make all_pref follow cleaned prefetcher names
    df.all_pref = list(zip(df.L1D_pref, df.L2C_pref, df.LLC_pref))
    return df

def mean(values: np.ndarray, metric: str, 
         weights: Optional[np.ndarray] = None):
    """Compute the mean of a particular metric, with the right formula.

    Parameters:
        values: A list of values.
        metric: A string of the metric, for selecting the right formula.
        weights: A list of weights, for weighted averages.

    Returns:
        mean: The mean of the values.    
    """
    if type(weights) is np.ndarray:
        assert np.isclose(np.sum(weights), 1), 'Weights should sum to 1'
    # ipc_improvement: gmean
    if 'ipc_improvement' in metric:
        # Add 100 to prevent negative values (so that 100 = no prefetcher baseline)
        return stats.gmean(values + 100, weights=weights) - 100
    # mpki_reduction: gmean
    if 'mpki_reduction' in metric:
        # Take gmean of relative misses instead of MPKI reduction to prevent negative values
        return 100 - stats.gmean(100 - values, weights=weights)
    # Others: amean
    else:
        return np.average(values, weights=weights)
