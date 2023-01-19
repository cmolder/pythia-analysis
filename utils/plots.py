from itertools import product
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import utils, stats


def plot_metric(data_df: dict, metric: str,
                suite: str = 'spec06',
                phase: str = 'one_phase',
                # Matplotlib
                figsize: Tuple[int, int] = None,
                dpi: int = None,
                legend: bool = True,
                suptitle: Optional[str] = None,
                colors: dict = {},
                annotations: dict = {},
                legend_kwargs: dict = {},
                label_kwargs: dict = {}):
    """Plot a specific metric for different prefetchers within a suite.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        metric: The metric to plot.
        suite: The name of the suite.
        dpi: The matplotlib DPI.
        figsize: The matplotlib figsize.
        

    Returns: None
    """
    def match_prefetcher(x): return x != (
        'no', 'no', 'no')  # Used in p_samples

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    num_samples = len(data_df.items())
    gap = num_samples + 1

    traces = list(list(data_df.values())[0].cpu0_trace.unique())

    min_y, max_y = 0, 0
    for i, (setup, df) in enumerate(data_df.items()):
        for j, tr in enumerate(traces):
            rows = df[df.cpu0_trace == tr]
            pos = (gap * j) + i
            #print(f'[DEBUG] i={i} j={j} setup={setup} tr={tr} pos={pos}, {pos+1}')
            p_samples = (rows[rows.all_pref.apply(match_prefetcher)])
            if p_samples.empty:
                p_min, p_mean, p_max = np.nan, np.nan, np.nan
            else:
                p_min = p_samples[metric].min()
                p_mean = p_samples[metric].mean(),
                p_max = p_samples[metric].max()
                max_y = max(max_y, p_max)
                min_y = min(min_y, p_min)

            #print(f'[DEBUG] {tr} Regular {setup} {p_mean:.2f} {p_min:.2f} {p_max:.2f}')
            # Plot bar
            ax.bar(pos, p_mean,
                   label=f'{setup}' if j == 0 else None,
                   color=colors[setup] if setup in colors.keys() else f'C{i}')

            # Plot bar error handles
            # ax.errorbar(pos, p_mean,
            #             yerr=[[p_mean - p_min], [p_max - p_mean]],
            #             color='black')

            # Plot bar annotations
            # TODO: Use a better scheme for annotation keys
            annotations_key = p_samples.cpu0_full_trace.iloc[0]
            if setup in annotations and annotations_key in annotations[setup]:
                ax.annotate(annotations[setup][annotations_key], xy=(pos, p_max), ha='center', va='bottom', size=8)

    ax.set_xlim(-gap/2, len(traces) * gap + gap/2)
    ax.set_xticks(np.arange(0, len(traces)) * gap + (num_samples / 2 - 0.5))
    ax.set_xticklabels(traces, **label_kwargs)
    ax.set_xlabel('Trace')

    # Set ticks based on metric
    tick_gaps = {
        'ipc_improvement': 10,
        'accuracy': 10,
        'coverage': 10,
        'mpki_reduction': 10
    }
    round_to_multiple = lambda num, mul : mul * round(num / mul)
    for metric_type, gap in tick_gaps.items():
        if metric_type in metric:
            ax.set_yticks(np.arange(
                round_to_multiple(min_y, gap), 
                round_to_multiple(max_y, gap) + gap, 
                gap))

    ax.set_ylabel(metric.replace('_', ' '))
    ax.grid(axis='y', color='lightgray')


    if legend:
        ax.legend(**legend_kwargs)  # bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    title = f'{metric.replace("_", " ")} ({suite} {phase})'
    if suptitle:
        title = f'{suptitle} {title}'
    fig.suptitle(title)
    fig.tight_layout()


def plot_everything(data_df: Dict[str, pd.DataFrame],
                    suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                    metrics: List[str] = ['ipc_improvement'],
                    **kwargs):
    """Plot multiples metrics for different prefetchers across suites.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        suites: A dict of suite names and suites.
        metrics: A list of metrics.
        figsize: The matplotlib figsize.
        dpi: The matplotlib DPI.

    Returns: None
    """
    for suite, phase in suites:
        data_df_ = {k: v[v.cpu0_trace.isin(utils.suites[suite])]
                    for k, v in data_df.items()}
        # Compute phase traces
        phase_simpoints = []
        for k, v in utils.phases[phase].items():
            phase_simpoints.append(f'{k}_{v}' if v != 'default' else k)

        for k, v in data_df_.items():
            v = v[v.cpu0_full_trace.isin(ps for ps in phase_simpoints)]
            v = stats.add_means(v)  # Add mean as an extra trace
            v = v.set_index('run_name')
            data_df_[k] = v

        print(f'=== {suite} {phase} ===')
        for metric in metrics:
            plot_metric(data_df_, metric, suite, phase, **kwargs)
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

    # Used in p_samples
    def match_prefetcher(x): 
        return x != ('no', 'no', 'no')

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    num_samples = len(data_df.items())
    gap = num_samples + 1

    max_y = 0
    for i, (setup, df) in enumerate(data_df.items()):
        #df = df[df.pythia_level_threshold == float('-inf')]
        df = df[df.cpu0_trace == benchmark]

        pos = gap + i
        p_samples = (df[df.all_pref.apply(match_prefetcher)][metric])
        p_mean = p_samples.mean()
        p_max = p_samples.max()
        max_y = max(max_y, p_max)
        color = f'C{i}'
        ax.bar(pos, p_mean, label=f'{setup}', color=color)

    # Set ticks based on metric
    if any(s in metric for s in ['ipc_improvement', 'accuracy', 'coverage', 'mpki_reduction']):
        #max_y = 150
        ax.set_yticks(np.arange(0, round(max_y, -1) + 10, 10))

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
        data_df_ = {k: v[v.cpu0_trace == benchmark] for k, v in data_df.items()}
        print(benchmark)
        for metric in metrics:
            plot_metric_benchmark(data_df_, benchmark, metric,
                                  figsize=figsize, dpi=dpi)
            plt.show()
