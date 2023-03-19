from itertools import product
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import utils, tables


"""
Plotting helpers
"""
def metric_tick_distances(metric: str) -> Optional[int]:
    metric = metric.lower()
    if 'ipc_improvement' in metric:
        return 10 #5
    elif 'accuracy' in metric:
        return 10
    elif 'coverage' in metric:
        return 10
    else:
        return None

"""
Plotting functions
"""
def plot_metric(table: pd.DataFrame,
                metric: Optional[str] = "",
                suite: Optional[str] = "",
                phase: Optional[str] = "",
                secondary_table: Optional[pd.DataFrame] = None,
                # Plotting kwargs
                figsize: Tuple[int, int] = None,
                dpi: int = None,
                legend: bool = True,
                suptitle: Optional[str] = "",
                colors: dict = {},
                legend_kwargs: dict = {},
                label_kwargs: dict = {},
                ylim: Optional[Tuple[float, float]] = None,):
    """Plot a specific metric for different prefetchers within a suite.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        metric: The metric to plot.
        suite: The name of the suite.
        dpi: The matplotlib DPI.
        figsize: The matplotlib figsize.
        
    Returns: None
    """
    Xs, ys, ys_secondary = {}, {}, {}
    min_y, max_y = 0.0, 0.0
    num_entries, num_traces = len(table), len(table.columns)
    gap, margin = 2, 2
    for i, (index, row) in enumerate(table.iterrows()):
        Xs[index] = (np.arange(num_traces) * (num_entries + gap)) + i
        ys[index] = row.values
        min_y = min(min_y, min(row.values))
        max_y = max(max_y, max(row.values))
    if secondary_table is not None:
        for i, (index, row) in enumerate(secondary_table.iterrows()):
            ys_secondary[index] = row.values
            min_y = min(min_y, min(row.values))
            max_y = max(max_y, max(row.values))

    # Bars
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for k, x in Xs.items():
        if k in ys_secondary:
            ax.bar(x, ys_secondary[k], color='lightgray')
        ax.bar(x, ys[k], label=k, color=colors[k])

    # X-axis
    ax.set_xlim(-margin, (num_traces * (num_entries + gap) - gap) + margin - 1)
    ax.set_xticks(np.arange(num_traces) * (num_entries + gap) + (num_entries - 1) / 2)
    ax.set_xticklabels(table.columns, **label_kwargs)
    ax.set_xlabel('Benchmark')

    # Y-axis
    ytick_dist = metric_tick_distances(metric)
    ylim_lower, ylim_upper = (min_y, max_y) if ylim is None else (ylim[0], ylim[1])
    if ytick_dist is not None:
        round_to_multiple = lambda num, mul : mul * round(num / mul)
        ax.set_yticks(np.arange(round_to_multiple(ylim_lower, ytick_dist), round_to_multiple(ylim_upper, ytick_dist) + 1, ytick_dist))
    ax.set_ylim(None if ylim is None else ylim_lower, None if ylim is None else ylim_upper)
    ax.set_ylabel(metric.replace('_', ' '))

    # Grid
    ax.grid(axis='y', color='lightgray')
    ax.set_axisbelow(True)

    # Legend
    if legend:
        ax.legend(**legend_kwargs)

    # Title
    title = f'{suptitle} {metric.replace("_", " ")}'
    if suite != '' and phase != '':
        title += f' ({suite} {phase})'
    elif suite != '':
        title += f' ({suite})'

    fig.suptitle(title)
    fig.tight_layout()


"""
Plotting function wrappers
"""
def plot_everything(data_df: Dict[str, pd.DataFrame],
                    suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                    metrics: List[str] = ['ipc_improvement'],
                    **kwargs):
    """Plot multiple metrics for schemes across multiple suites.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        suites: A dict of suite names and suites.
        metrics: A list of metrics.
        figsize: The matplotlib figsize.
        dpi: The matplotlib DPI.

    Returns: None
    """
    for suite, phase in suites:
        print(f'=== {suite} {phase} ===')
        for metric in metrics:
            table = tables.gen_table_metric(data_df, suite, phase, metric).T
            table.columns = [utils.clean_trace_name(tr, metric=metric) for tr in table.columns]

            # Plot results
            plot_metric(table, metric, suite, phase, **kwargs)
            plt.show()


def plot_everything_accuracy(data_df: Dict[str, pd.DataFrame],
                             suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                             level='L2C',
                             **kwargs):
    """Plot (un)timely accuracy for different schemes across multiple suites.
    """
    for suite, phase in suites:
        metric = f'{level}_timely_accuracy'
        secondary_metric = f'{level}_accuracy'

        table = tables.gen_table_metric(data_df, suite, phase, metric).T
        secondary_table = tables.gen_table_metric(data_df, suite, phase, secondary_metric).T

        table.columns = [utils.clean_trace_name(tr, metric=metric) for tr in table.columns]
        secondary_table.columns = [utils.clean_trace_name(tr, metric=secondary_metric) for tr in secondary_table.columns]

        # Plot results
        kwargs['ylim'] = (0.0, 100.0)
        plot_metric(table, f'{level} accuracy', suite, phase, secondary_table, **kwargs)
        plt.show()


def plot_everything_coverage(data_df: Dict[str, pd.DataFrame],
                             suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                             level='L2C',
                             **kwargs):
    """Plot (un)timely coverage for different schemes across multiple suites.
    """
    for suite, phase in suites:
        metric = f'{level}_coverage'
        secondary_metric = f'{level}_untimely_coverage'

        table = tables.gen_table_metric(data_df, suite, phase, metric).T
        secondary_table = tables.gen_table_metric(data_df, suite, phase, secondary_metric).T

        table.columns = [utils.clean_trace_name(tr, metric=metric) for tr in table.columns]
        secondary_table.columns = [utils.clean_trace_name(tr, metric=secondary_metric) for tr in secondary_table.columns]

        # Plot results
        kwargs['ylim'] = (0.0, 100.0)
        plot_metric(table, f'{level} coverage', suite, phase, secondary_table, **kwargs)
        plt.show()


def plot_metric_means(data_df: Dict[str, pd.DataFrame],
                      suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                      metric: str = ['ipc_improvement'],
                      **kwargs):
    """Plot the mean of a metric across multiple suites, and overall.

    The "all" mean is weighted by the number of benchmarks in each
    suite, as defined in utils.suites.

    Parameters:
        data_df: A dict of prefetchers and their statistics dataframes.
        suites: A dict of suite names and suites.
        metrics: A list of metrics.
        kwargs: Plotting kwargs.

    Returns: None
    """
    table = tables.gen_table_metric_all(data_df, suites, metric)
    metric = f'{utils.get_mean_type(metric)} {metric}'
    plot_metric(table, metric=metric, **kwargs)