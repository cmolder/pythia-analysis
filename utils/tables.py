from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from IPython.display import display

from utils import stats, utils


def table_suite(data_df: Dict[str, pd.DataFrame],
                suite: List[str] = utils.spec06,
                metrics: List[str] = ['ipc_improvement']):
    """Summarize statsitics on a single suite.

    Parameters:
        data_df: A dictionary of prefetchers and their statistics dataframes.
        suite: A list of benchmarks in the suite to include.
        metrics: A list of metrics to include.

    Returns: None
    """
    data_df_ = {k: v[v.trace.isin(suite)] for k, v in data_df.items()}
    for k, v in data_df_.items():
        v = stats.add_means(v)  # Add mean as an extra trace
        v = v.set_index('trace').round(2)
        print(k)
        display(v[metrics])


def table_everything(data_df: Dict[str, pd.DataFrame],
                     suites: Dict[str, List[str]] = {'SPEC 06': utils.spec06},
                     metrics: List[str] = ['ipc_improvement']):
    """Summarize statsitics on multiple suites.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        suites: A dict of suite names and their lists of benchmarks.
        metrics: A list of metrics to include.

    Returns: None
    """
    for suite_name, suite in suites.items():
        print(f'=== {suite_name} ===')
        table_suite(data_df, suite, metrics)


def load_stats_csv(stats_csv: str,
                   prefetchers: List[str],
                   prefetchers_level = 'l2',
                   separate_degrees: bool = False,
                   seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
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
    df = utils.read_data_file(stats_csv)
    if seed is not None:
        df = df[df.seed == seed]
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


def load_stats_csv_pythia(stats_csv: str,
                          feature_sets: List[Set],
                          seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
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
    if seed is not None:
        df = df[df.seed == seed]
    if feature_sets == []:
        feature_sets = df.pythia_features.unique()
        feature_sets = [value_to_set(s) for s in feature_sets]
    for feat_set in feature_sets:
        data_df[set_to_string(feat_set)] = (
            df[df.pythia_features.apply(value_to_set) == feat_set])
    return data_df
