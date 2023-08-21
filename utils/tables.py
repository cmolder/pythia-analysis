import os
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy
import pandas as pd
from IPython.display import display

from utils import stats, utils

def gen_table_suite(data_df: Dict[str, pd.DataFrame],
                    suite: str = 'spec06',
                    phase: str = 'one_phase',
                    metrics: List[str] = ['ipc_improvement']) -> Dict[str, pd.DataFrame]:
    """Summarize statsitics on a single suite.

    Parameters:
        data_df: A dictionary of prefetchers and their statistics dataframes.
        suite: A list of benchmarks in the suite to include.
        phase: Which phase to consider in each benchmark.
        metrics: A list of metrics to include.

    Returns: None

    TODO: Handle multicore
    """
    tables = {}
    data_df_ = {k: v[v.cpu0_trace.isin(utils.suites[suite])] 
                for k, v in data_df.items()}

    for k, v in data_df_.items():
        v = v[v.cpu0_simpoint.isin(v for v in utils.phases[phase].values())]
        v = stats.add_means(v)  # Add mean as an extra trace
        v = v.set_index('run_name')
        tables[k] = v[metrics]

    return tables


def table_suite(data_df: Dict[str, pd.DataFrame],
                suite: str = 'spec06',
                phase: str = 'one_phase',
                metrics: List[str] = ['ipc_improvement']):
    """Summarize statsitics on a single suite.

    Parameters:
        data_df: A dictionary of prefetchers and their statistics dataframes.
        suite: A list of benchmarks in the suite to include.
        phase: Which phase to consider in each benchmark.
        metrics: A list of metrics to include.

    Returns: None

    TODO: Handle multicore
    """
    tables = gen_table_suite(data_df, suite, phase, metrics)
    for k, v in tables.items():
        print(k)
        display(v)

def process_run_names(run_names: pd.Series):
    run_names = run_names.copy()
    run_phases = run_names.copy()

    for i, rn in run_names.iteritems():
        rn = str(rn).split('_')
        if rn[0] == 'mix':
            run_names[i] = f'mix{rn[1]}'
            run_phases[i] = 'default'
        elif len(rn) == 1:
            run_names[i] = rn[0]
            run_phases[i] = 'default'
        else:
            run_names[i] = '_'.join(rn[:-1])
            run_phases[i] = rn[-1]

    return run_names, run_phases


def gen_table_metric(data_df: Dict[str, pd.DataFrame],
                     suite: str = 'spec06',
                     phase: str = 'one_phase',
                     metric: str = 'ipc_improvement',
                     add_mean: bool = True) -> pd.DataFrame:
    """Summarize statsitics on a single metric.

    Parameters:
        data_df: A dictionary of prefetchers and their statistics dataframes.
        suite: A list of benchmarks in the suite to include.
        phase: Which phase to consider in each benchmark.
        metric: A metric to include.

    Returns: None
    """
    data_df_ = {k: v.copy() for k, v in data_df.items()}
    
    for k, v in data_df_.items():
        run_names, run_phases = process_run_names(v.run_name)
        v['run_name_adj'] = run_names
        v['run_phase_adj'] = run_phases

        # Filter on this suite's runs
        v = v[v.run_name_adj.isin(utils.suites[suite])]

        # Filter on this phase's runs
        v = v[v.apply(lambda row : row.run_phase_adj == utils.phases[phase][row.run_name_adj], axis=1)]

        # Add mean as an extra row
        if add_mean:
            v = stats.add_means(v)
        
        # Cleanup, re-assign the value
        v = v.set_index('run_name')[metric]
        v.name = k
        data_df_[k] = v
    
    metric_df = pd.concat(data_df_.values(), axis=1)
    return metric_df


def table_metric(data_df: Dict[str, pd.DataFrame],
                 suite: str = 'spec06',
                 phase: str = 'one_phase',
                 metric: str = 'ipc_improvement'):
    """Summarize statsitics on a single metric.

    Parameters:
        data_df: A dictionary of prefetchers and their statistics dataframes.
        suite: A list of benchmarks in the suite to include.
        phase: Which phase to consider in each benchmark.
        metric: A metric to include.

    Returns: None
    """
    display(gen_table_metric(data_df, suite, phase, metric))


def gen_table_metric_all(data_df: Dict[str, pd.DataFrame],
                        suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                        metric: str = 'ipc_improvement') -> pd.DataFrame:
    """Summarize statistics on a single metric, across multiple
    suites. The mean is weighted evenly per-benchmark.
    """
    tables = {s: gen_table_metric(data_df, s, p, metric) for s, p in suites}

    # Get the means for each suite
    means = {}
    weights = {'all': 0}
    for suite, table in tables.items():
        means[suite] = table[table.index == 'mean'].copy().reset_index()
        means[suite].loc[:, 'suite'] = suite
        means[suite].set_index('suite', inplace=True)
        del means[suite]['run_name']
        means[suite] = means[suite].squeeze()
        weights[suite] = len(utils.suites[suite])
        
    means_df = pd.DataFrame(means.values()).T
    
    # Add an overall mean weighted by the number of benchmarks in each suite.
    # Create the "all" column and move it to the front
    means_df.loc[:, 'all'] = 0.0 # To be filled
    all_col = means_df.pop('all')
    means_df.insert(0, 'all', all_col)

    # Fill in the values of the "all" column
    for index, row in means_df.iterrows():
        row_weights = np.array([weights[i] for i, _ in row.iteritems()])
        row_weights = row_weights / np.sum(row_weights) # Normalize

        if index == "Bingo":
            print(row_weights.tolist())
        row_values = np.array([v for _, v in row.iteritems()])
        if index == "Bingo":
            print(row_values.tolist())
        row_mean = utils.mean(row_values, metric, row_weights)
        means_df.loc[index, 'all'] = row_mean

    means_df = means_df.round(6)
    return means_df


def table_metric_all(data_df: Dict[str, pd.DataFrame],
                     suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                     metric: str = 'ipc_improvement'):
    """Summarize statistics on a single metric, across multiple
    suites. The mean is weighted evenly per-benchmark.
    """
    display(gen_table_metric_all(data_df, suites, metric))


def table_everything(data_df: Dict[str, pd.DataFrame],
                     suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                     metrics: List[str] = ['ipc_improvement']):
    """Summarize statsitics on multiple suites.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        suites: A dict of suite names and their lists of benchmarks.
        metrics: A list of metrics to include.

    Returns: None
    """
    # for suite, phase in suites:
    #     print(f'=== {suite} {phase} ===')
    #     table_suite(data_df, suite, phase, metrics)
    for suite, phase in suites:
        print(f'=== {suite} {phase} ===')
        for metric in metrics:
            print(metric)
            table_metric(data_df, suite, phase, metric)


def load_stats_csv(base_dir: str,
                   stats_csv: str,
                   prefetchers: List[str],
                   prefetchers_level = 'l2',
                   separate_degrees: bool = False) -> Dict[str, pd.DataFrame]:
    """Load stats for arbitrary prefetchers.

    Parameters:
        stats_csv: Path to the stats .csv file, as generated by the
            evaluate script in Pythia/experiments.
        prefetchers: A list of prefetchers to include
        separate_degrees: If passed, break down the prefetchers by degree.
            For example, ISB_real with degree 2 will become isb_real_2.
        seed: The ChampSim/Pythia seed to use. If not provided, assume
            the stats only consider one seed.

    Returns:
        data_df: A dict of prefetchers and their statistics dataframes.
    """
    stats_csv = os.path.join(base_dir, stats_csv)
    df = utils.read_data_file(stats_csv)
    df.fillna(0, inplace=True)

    data_df = {}
    if prefetchers == []:
        prefetchers = df[f'{prefetchers_level.upper()}_pref'].unique()

    def get_all_pref_key(pf, level):
        if level == 'l1d':
            return (pf, 'no', 'no')
        elif level == 'l2':
            return ('no', pf, 'no')
        else:
            return ('no', 'no', pf)


    if separate_degrees:
        deg_key = f'{prefetchers_level.upper()}_pref_degree'
        for pf in prefetchers:
            df_ = df[df.all_pref == get_all_pref_key(pf, prefetchers_level)]
            for d in df_[deg_key].unique():
                deg_key_mod = d.replace('(', '').replace(')', '')
                deg_key_mod = deg_key_mod.rstrip(',')
                deg_key_mod = f'{pf}_{deg_key_mod}'
                data_df[deg_key_mod] = df_[df_[deg_key] == d]

    else:
        for pf in prefetchers:
            data_df[pf] = (
                df[df.all_pref == get_all_pref_key(pf, prefetchers_level)])

    return data_df


def merge_best_prefetcher(*dfs, metric='ipc', method='max'):
    """Produce a best prefetcher on each trace, by picking the "best"
    among the provided prefetcher dfs on a metric and method (min/max)
    
    Parameters: TODO
    """
    df_all = pd.concat(dfs)
    if method == 'max':
        return (df_all.sort_values(metric, ascending=False) # max
                      .drop_duplicates('full_trace')
                      .sort_values('full_trace', ascending=True))
                      
    else:
        return (df_all.sort_values(metric, ascending=True) # min
                      .drop_duplicates('full_trace') 
                      .sort_values('full_trace', ascending=True))


def load_stats_csv_pythia(base_dir: str,
                          stats_csv: str,
                          feature_sets: List[Set],
                          feature_key = 'pythia_features') -> Dict[str, pd.DataFrame]:
    """Load stats for specific Pythia feature sets.

    Parameters:
        stats_csv: Path to the stats .csv file, as generated by the 
            evaluate script in Pythia/experiments.
        feature_sets: A list of string sets that represent the Pythia
            feature sets to store (e.g. [{'Delta_Path', 'PC_Delta'}]).
        seed: The ChampSim/Pythia seed to use. If not provided, assume
            the stats only consider one seed.

    Returns:
        data_df: A dict of prefetchers and their statistics dataframes.
    """
    stats_csv = os.path.join(base_dir, stats_csv)

    # Convert string/tuple/pd entry to unordered set.
    def value_to_set(value):
        if pd.notna(value):
            return set(eval(value))
        return set()

    # Convert set to string.
    def set_to_string(set):
        return ', '.join(f for f in sorted(set))

    data_df = {}
    df = utils.read_data_file(stats_csv)
    if feature_sets == []:
        feature_sets = df[feature_key].unique()
        feature_sets = [value_to_set(s) for s in feature_sets]
    for feat_set in feature_sets:
        data_df[set_to_string(feat_set)] = (
            df[df[feature_key].apply(value_to_set) == feat_set])
    return data_df


def load_stats_csv_next_line(base_dir: str, 
                             stats_csv: str,
                             offsets: Optional[List[Set]] = None) -> Dict[str, pd.DataFrame]:
    """Load stats for specific fixed-offsets.

    TODO: docstring
    """
    stats_csv = os.path.join(base_dir, stats_csv)

    data_df = {}
    df = utils.read_data_file(stats_csv)
    if offsets is None:
        offsets = df.next_line_offset.unique()
    for offset in offsets:
        data_df[offset] = df[df.next_line_offset == str(offset)]
    return data_df
