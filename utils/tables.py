from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from IPython.display import display

from utils import utils


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
                   seed: Optional[int] = None) -> Dict[Tuple[str, ...], pd.DataFrame]:
    """Load stats for arbitrary prefetchers.

    Parameters:
        stats_csv: Path to the stats .csv file, as generated by the 
            evaluate script in Pythia/experiments.
        prefetchers: A list of L2 prefetchers to include
            (TODO: Support other levels of the cache).
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
        prefetchers = df.L2C_pref.unique()
    for pf in prefetchers:
        data_df[pf] = df[df.all_pref == ('no', pf, 'no')]
    return data_df


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
    def string_to_set(r): return set(
        eval(r))  # Convert string/tuple to unordered set.
    # Convert set to string.
    def set_to_string(s): return ', '.join(f for f in sorted(s))
    data_df = {}
    df = utils.read_data_file(stats_csv)
    if seed is not None:
        df = df[df.seed == seed]
    if feature_sets == []:
        feature_sets = df.pythia_features.unique()
        feature_sets = [string_to_set(s) for s in feature_sets]
    for feat_set in feature_sets:
        data_df[set_to_string(feat_set)] = (
            df[df.pythia_features.apply(string_to_set) == feat_set])
    return data_df
