from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import utils, stats


def plot_metric(data_df: dict, metric: str,
                figsize: Tuple[int, int] = None,
                dpi: int = None,
                suite_name: str = ''):
    """Plot a specific metric for different prefetchers within a suite.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        metric: The metric to plot.
        dpi: The matplotlib DPI.
        figsize: The matplotlib figsize.
        suite_name: The name of the suite, added to the plot title.

    Returns: None
    """
    def match_prefetcher(x): return x != (
        'no', 'no', 'no')  # Used in p_samples

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    num_samples = len(data_df.items())
    gap = num_samples + 1

    traces = list(list(data_df.values())[0].trace.unique())
    trace_names = traces + \
        (['amean'] if metric in utils.amean_metrics else ['gmean'])
    traces = traces + ['mean']

    largest_y = 0
    for i, (setup, df) in enumerate(data_df.items()):
        df = df[df.pythia_level_threshold == float('-inf')]
        df = stats.add_means(df)  # Add mean as an extra trace
        for j, tr in enumerate(traces):
            rows = df[df.trace == tr]
            pos = (gap * j) + i
            #print(f'[DEBUG] i={i} j={j} setup={setup} tr={tr} pos={pos}, {pos+1}')
            p_samples = (rows[rows.all_pref.apply(match_prefetcher)][metric])
            #p_min = p_samples.min()
            p_mean = p_samples.mean(),
            p_max = p_samples.max()
            largest_y = max(largest_y, p_max)

            #print(f'[DEBUG] {tr} Regular {setup} {p_mean:.2f} {p_min:.2f} {p_max:.2f}')
            ax.bar(pos, p_mean,
                   label=f'{setup}' if j == 0 else None,
                   color=f'C{i}')
            # ax.errorbar(pos, p_mean,
            #             yerr=[[p_mean - p_min], [p_max - p_mean]],
            #             color='black')

    ax.set_xticks(np.arange(0, len(df.trace.unique())) * gap + (num_samples/2))
    ax.set_xticklabels(trace_names, rotation=90)
    ax.set_xlabel('Trace')

    # Set ticks based on metric
    TICK_METRICS = ['ipc_improvement',
                    'accuracy', 'coverage', 'mpki_reduction']
    if any(s in metric for s in TICK_METRICS):
        #largest_y = 150
        ax.set_yticks(np.arange(0, round(largest_y, -1) + 10, 10))

    ax.set_ylabel(metric.replace('_', ' '))
    ax.grid(axis='y', color='lightgray')
    ax.set_axisbelow(True)

    fig.legend()  # bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    fig.suptitle(f'{metric.replace("_", " ")} ({suite_name})')
    fig.tight_layout()


def plot_everything(data_df: Dict[str, pd.DataFrame],
                    suites: Dict[str, List[str]] = {'SPEC 06': utils.spec06},
                    metrics: List[str] = ['ipc_improvement'],
                    figsize: Tuple[int, int] = (9, 5),
                    dpi: Optional[int] = None):
    """Plot multiples metrics for different prefetchers across suites.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        suites: A dict of suite names and suites.
        metrics: A list of metrics.
        figsize: The matplotlib figsize.
        dpi: The matplotlib DPI.

    Returns: None
    """
    for suite_name, suite in suites.items():
        data_df_ = {k: v[v.trace.isin(suite)] for k, v in data_df.items()}
        print(f'=== {suite_name} ===')
        for metric in metrics:
            plot_metric(data_df_, metric,
                        suite_name=suite_name, figsize=figsize, dpi=dpi)
            plt.show()


def plot_metric_benchmark(data_df: dict, benchmark: str, metric: str,
                          figsize: Tuple[int, int] = None,
                          dpi: Optional[int] = None):
    """Plot a specific metric for different prefetchers on one benchmark.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        benchmark: The benchmark to consider.
        metric: The metric to plot.
        dpi: The matplotlib DPI.
        figsize: The matplotlib figsize.

    Returns: None
    """
    def match_prefetcher(x): return x != (
        'no', 'no', 'no')  # Used in p_samples

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    num_samples = len(data_df.items())
    gap = num_samples + 1

    largest_y = 0
    for i, (setup, df) in enumerate(data_df.items()):
        df = df[df.pythia_level_threshold == float('-inf')]
        df = df[df.trace == benchmark]

        pos = gap + i
        p_samples = (df[df.all_pref.apply(match_prefetcher)][metric])
        p_mean = p_samples.mean()
        p_max = p_samples.max()
        largest_y = max(largest_y, p_max)
        color = f'C{i}'
        ax.bar(pos, p_mean, label=f'{setup}', color=color)

    # Set ticks based on metric
    if any(s in metric for s in ['ipc_improvement', 'accuracy', 'coverage', 'mpki_reduction']):
        #largest_y = 150
        ax.set_yticks(np.arange(0, round(largest_y, -1) + 10, 10))

    ax.set_xticks([])
    ax.set_xlabel('Prefetcher')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.grid(axis='y', color='lightgray')
    ax.set_axisbelow(True)

    fig.legend()  # bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    fig.suptitle(f'{metric.replace("_", " ")} ({benchmark})')
    fig.tight_layout()


def plot_everything_benchmark(data_df: Dict[str, pd.DataFrame],
                              benchmarks: List[str],
                              metrics: List[str] = ['ipc_improvement'],
                              figsize: Tuple[int, int] = (9, 5),
                              dpi: Optional[int] = None):
    """Plot benchmark graphs for different prefetchers across metrics.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        benchmarks: A list of benchmarks.
        metrics: A list of metrics.
        figsize: The matplotlib figsize.
        dpi: The matplotlib DPI.

    Returns: None
    """
    for benchmark in benchmarks:
        data_df_ = {k: v[v.trace == benchmark] for k, v in data_df.items()}
        print(benchmark)
        for metric in metrics:
            plot_metric_benchmark(data_df_, benchmark, metric,
                                  figsize=figsize, dpi=dpi)
            plt.show()
