from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from plot_utils import collate
from naboo_utils.file import avg_fn_wrapper


"""
Plotting helpers
"""
def metric_tick_distances(metric: str) -> Optional[int]:
    """Return the tick distance for a metric, if not provided by user.
    """
    metric = metric.lower()
    if 'ipc_improvement' in metric:
        return 10
    elif 'accuracy' in metric:
        return 10
    elif 'coverage' in metric:
        return 10
    else:
        return None
    
def get_mean_string(statistic_name: str):
    """Infer the string for a mean statistic based on the
    statistic name."""
    if "gmean" in statistic_name:
        return "gmean"
    elif "hmean" in statistic_name:
        return "hmean"
    else:
        return "amean"
    
def get_mean(statistic_name: str, table: pd.DataFrame):
    """Infer the mean for a statistic based on the
    statistic name."""
    mean_string = get_mean_string(statistic_name)
    if mean_string == "gmean":
        fn = stats.gmean
    elif mean_string == "hmean":
        fn = stats.hmean
    else:
        fn = np.average

    return table.apply(lambda row : avg_fn_wrapper(fn, mean_string, row, statistic_name.split(".")), axis=1)

def get_benchmark_table(experiments: Dict[str, collate.ExperimentCollator],
                        suite_name: str,
                        statistic_name: str,
                        benchmarks: Optional[List[str]] = None,
                        add_mean: bool = True) -> pd.DataFrame:
    """Get the table for a benchmark, with some extra processing
    for plotting.
    """
    table = collate.get_benchmark_statistic(experiments, suite_name, statistic_name).T
    if benchmarks is not None:
        table = table[benchmarks]
    if add_mean:
        mean = get_mean(statistic_name, table)
        table[get_mean_string(statistic_name)] = mean
    return table

def get_suite_table(experiments: Dict[str, collate.ExperimentCollator],
                    statistic_name: str,
                    suites: Optional[List[str]] = None,
                    add_mean: bool = True) -> pd.DataFrame:
    """Get the table for a suite, with some extra processing
    for plotting.
    """
    table = collate.get_suite_statistic(experiments, statistic_name).T
    if suites is not None:
        table = table[suites]
    if add_mean:
        mean = get_mean(statistic_name, table)
        table[get_mean_string(statistic_name)] = mean
    return table

def plot_table(table: pd.DataFrame,
               secondary_table: Optional[pd.DataFrame] = None,
               # Plotting kwargs
               figsize: Tuple[int, int] = None,
               dpi: int = None,
               legend: bool = True,
               add_suptitle: bool = True, suptitle: str = "",
               add_xlabel: bool = True, xlabel: str = "Benchmark",
               add_ylabel: bool = True, ylabel: str = "",
               colors: defaultdict = defaultdict(lambda: None),
               secondary_colors: List[str] = ["lightgray"],
               hatches: defaultdict = defaultdict(lambda: None),
               annotations: defaultdict = defaultdict(str),
               legend_kwargs: dict = {},
               label_kwargs: dict = {},
               ylim: Optional[Tuple[float, float]] = None,
               ytick_distance: Optional[int] = None):
    """Plot a statistic for different experiments.

    Parameters:
        table: A table of metrics.
        secondary_table: A table of secondary metrics. (TODO: Support mutliple secondary tables)

    Plotting kwargs:
        figsize: The matplotlib figsize.
        dpi: The matplotlib DPI.
        TODO: Describe the rest
    """
    
    Xs, ys, ys_secondary = {}, {}, {}
    min_y, max_y = 0.0, 0.0
    num_entries, num_traces = len(table), len(table.columns)
    gap, margin = 2, 2

    # Plot
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
            ax.bar(x, ys_secondary[k], color=secondary_colors[0], 
                   width=1.0, edgecolor='gray', linewidth=0.3, 
                   zorder=0.99)
        ax.bar(x, ys[k], label=k, color=colors[k], hatch=hatches[k], 
               edgecolor='black', linewidth=0.3, width=1.0)

        # Bar annotations
        for i, x in enumerate(x):
            trace = table.columns[i]
            if annotations[(k, trace)] != "":
                ax.annotate(annotations[(k, trace)], (x, 0), xytext=(x, 0), 
                            ha='center', va='bottom', rotation='90', fontsize=4)

    # X-axis
    ax.set_xlim(-margin, (num_traces * (num_entries + gap) - gap) + margin - 1)
    ax.set_xticks(np.arange(num_traces) * (num_entries + gap) + (num_entries - 1) / 2)
    ax.set_xticklabels(table.columns, **label_kwargs)
    if add_xlabel:
        ax.set_xlabel(xlabel)

    # Y-axis
    ylim_lower, ylim_upper = (min_y, max_y) if ylim is None else (ylim[0], ylim[1])
    if ytick_distance is not None:
        round_to_multiple = lambda num, mul : mul * round(num / mul)
        ax.set_yticks(np.arange(round_to_multiple(ylim_lower, ytick_distance), round_to_multiple(ylim_upper, ytick_distance) + 1, ytick_distance))
        ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(None if ylim is None else ylim_lower, None if ylim is None else ylim_upper)
    if add_ylabel:
        ax.set_ylabel(ylabel, fontsize=8)

    # Grid
    ax.grid(axis='y', color='darkgray', linewidth=0.3)
    ax.set_axisbelow(True)

    # Legend
    if legend:
        ax.legend(**legend_kwargs)

    # Title
    # TODO: Fix title
    # if add_suptitle:
    #     title = f'{suptitle} {metric.replace("_", " ")}'
    #     if suite != '' and phase != '':
    #         title += f' ({suite} {phase})'
    #     elif suite != '':
    #         title += f' ({suite})'
    #     fig.suptitle(title)
    
    fig.tight_layout()

"""
Plotting functions
"""
def plot_benchmark_statistic(experiments: Dict[str, collate.ExperimentCollator],
                             suite_name: str,
                             statistic_name: str,
                             secondary_statistic_name: Optional[str] = None,
                             benchmarks: Optional[List[str]] = None,
                             add_mean: bool = True,
                             **kwargs):
    """TODO: Docstring
    """
    # Gather table(s)
    table = get_benchmark_table(experiments, suite_name, statistic_name,
                                benchmarks=benchmarks, add_mean=add_mean)
    if secondary_statistic_name is not None:
        secondary_table = get_benchmark_table(experiments, suite_name,
                                              secondary_statistic_name,
                                              benchmarks=benchmarks,
                                              add_mean=add_mean)
    else:
        secondary_table = None

    plot_table(table, secondary_table, **kwargs)

def plot_suite_statistic(experiments: Dict[str, collate.ExperimentCollator],
                         statistic_name: str,
                         secondary_statistic_name: Optional[str] = None,
                         suites: Optional[List[str]] = None,
                         add_mean: bool = True,
                         **kwargs):
    """TODO: Docstring
    """
    table = get_suite_table(experiments, statistic_name,
                            suites=suites, add_mean=add_mean)
    if secondary_statistic_name is not None:
        secondary_table = get_suite_table(experiments, secondary_statistic_name, 
                                          suites=suites, add_mean=add_mean)
    else:
        secondary_table = None

    plot_table(table, secondary_table, **kwargs)

'''
def plot_overpredictions(cov_table: pd.DataFrame,
                         cov_late_table: pd.DataFrame,
                         overpred_table: pd.DataFrame,
                         suite: Optional[str] = "",
                         phase: Optional[str] = "",
                         level: Optional[str] = "L2C",
                         # Plotting kwargs
                         figsize: Tuple[int, int] = None,
                         dpi: int = None,
                         legend: bool = True,
                         add_mean: bool = True,
                         add_suptitle: bool = True, suptitle: str = "",
                         add_xlabel: bool = True, xlabel: str = "Benchmark",
                         colors: defaultdict = defaultdict(lambda: None),
                         hatches: defaultdict = defaultdict(lambda: None),
                         annotations: defaultdict = defaultdict(str),
                         legend_kwargs: dict = {},
                         label_kwargs: dict = {},
                         annotate_outliers: bool = True,
                         ymin: Optional[float] = None,
                         ymax: Optional[float] = None,
                         ytick_distance: Optional[int] = None):
    Xs, ys_cov, ys_cov_untimely, ys_over = {}, {}, {}, {}
    max_y = 0.0
    num_entries, num_traces = len(cov_table), len(cov_table.columns)
    gap, margin = 2, 2
    for i, (index, row) in enumerate(cov_table.iterrows()):
        Xs[index] = (np.arange(num_traces) * (num_entries + gap)) + i
        cov_row = row.values
        cov_untimely_row = cov_late_table.loc[index].values
        over_row = overpred_table.loc[index].values

        ys_cov[index] = cov_row
        ys_cov_untimely[index] = cov_untimely_row
        ys_over[index] = over_row

        # print(i, index, Xs)
        max_y = max(max_y, cov_row.max(), cov_untimely_row.max(), over_row.max() + 100.0)

    # Bars
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for i, (k, x) in enumerate(Xs.items()):
        # Timely coverage
        ax.bar(x, ys_cov[k], label=k, color=colors[k], hatch=hatches[k],
               edgecolor='black', linewidth=0.3, width=1.0)
        # Untimely coverage
        ax.bar(x, ys_cov_untimely[k], color='lightgray',zorder=0.99,
               edgecolor='black', linewidth=0.3, width=1.0) # hatch=hatches[k], 
        # Uncovered
        ax.bar(x, 100.0, color='darkgray', zorder=0.98,
               edgecolor='black', linewidth=0.3, width=1.0) # hatch=hatches[k],
        # Overpredicted
        ax.bar(x, ys_over[k], color='lightcoral', bottom=100.0, zorder=0.97,
               edgecolor='black', linewidth=0.3, width=1.0) #hatch=hatches[k], 

        # Annotate outliers
        max_bound = max_y if ymax is None else ymax
        bound = ys_over[k] + 100.0
        if annotate_outliers and (bound + 1 > max_bound).any(): 
            outlier_idxs = np.where(bound > max_bound)[0]
            for oi in outlier_idxs:
                oi_str = f'{bound[oi]:.0f}%\n({k})'
                ax.annotate(oi_str, xy=(x[oi] + 1, max_bound - 3), xytext=(x[oi] + 1, max_bound - 3), fontsize=6.75, ha='left', va='top')
            # print(k, ys_over[k])
            # print(outlier_idxs)

    # X-axis
    ax.set_xlim(-margin, (num_traces * (num_entries + gap) - gap) + margin - 1)
    ax.set_xticks(np.arange(num_traces) * (num_entries + gap) + (num_entries - 1) / 2)
    ax.set_xticklabels(cov_table.columns, **label_kwargs)
    if add_xlabel:
        ax.set_xlabel(xlabel)
    ax.axhline(100.0, color='black')

    # Y-axis
    ytick_dist = 20 if ytick_distance is None else ytick_distance
    ylim_lower = 0 if ymin is None else ymin
    ylim_upper = ytick_dist * math.ceil(max_y / ytick_dist) if ymax is None else ymax
    if ytick_dist is not None:
        round_to_multiple = lambda num, mul : mul * round(num / mul)
        ax.set_yticks(np.arange(round_to_multiple(ylim_lower, ytick_dist), round_to_multiple(ylim_upper, ytick_dist) + 1, ytick_dist))
        ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(ylim_lower, ylim_upper)
    ax.set_ylabel('Normalized accesses (%)', fontsize=8)

    # Grid
    ax.grid(axis='y', color='lightgray')
    ax.set_axisbelow(True)

    # Legend
    if legend:
        from matplotlib.patches import Patch
        custom_legend = [Patch(facecolor=colors[k], hatch=hatches[k], label=k) for k in cov_table.index]
        custom_legend.append(Patch(facecolor='lightgray', label='(Covered, late)'))
        custom_legend.append(Patch(facecolor='darkgray', label='(Not covered)'))
        custom_legend.append(Patch(facecolor='red', label='(Overpredicted)'))
        legend_kwargs['handles'] = custom_legend
        ax.legend(**legend_kwargs)

    fig.tight_layout()




"""
Plotting function wrappers
"""
def plot_everything(data_df: Dict[str, pd.DataFrame],
                    suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                    metrics: List[str] = ['ipc_improvement'],
                    add_mean: bool = True,
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
            table = tables.gen_table_metric(data_df, suite, phase, metric, add_mean).T
            table.columns = [utils.clean_trace_name(tr, metric=metric) for tr in table.columns]

            # Plot results
            plot_metric(table, metric, suite, phase, **kwargs)
            # plt.savefig('../../cvs/papers/micro-23r/figures/pythia_action_ordering.pdf')
            # plt.savefig('../../cvs/papers/micro-23r/figures/results_spec06.pdf')
            # 2/0
            # plt.show()


def plot_everything_accuracy(data_df: Dict[str, pd.DataFrame],
                             suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                             level: str = 'L2C',
                             add_mean: bool = True,
                             **kwargs):
    """Plot (un)timely accuracy for different schemes across multiple suites.
    """
    for suite, phase in suites:
        print(f'=== {suite} {phase} ===')
        metric = f'{level}_timely_accuracy'
        secondary_metric = f'{level}_accuracy'

        table = tables.gen_table_metric(data_df, suite, phase, metric, add_mean).T
        secondary_table = tables.gen_table_metric(data_df, suite, phase, secondary_metric, add_mean).T

        table.columns = [utils.clean_trace_name(tr, metric=metric) for tr in table.columns]
        secondary_table.columns = [utils.clean_trace_name(tr, metric=secondary_metric) for tr in secondary_table.columns]

        # Plot results
        kwargs['ylim'] = (0.0, 100.0)
        plot_metric(table, f'{level} accuracy', suite, phase, secondary_table, **kwargs)
        plt.show()


def plot_everything_coverage(data_df: Dict[str, pd.DataFrame],
                             suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                             level: str = 'L2C',
                             add_mean: bool = True,
                             **kwargs):
    """Plot (un)timely coverage for different schemes across multiple suites.
    """
    for suite, phase in suites:
        print(f'=== {suite} {phase} ===')
        metric = f'{level}_coverage'
        secondary_metric = f'{level}_untimely_coverage'

        table = tables.gen_table_metric(data_df, suite, phase, metric, add_mean).T
        secondary_table = tables.gen_table_metric(data_df, suite, phase, secondary_metric, add_mean).T

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


def plot_overprediction_means(data_df: Dict[str, pd.DataFrame],
                              level: str = 'L2C',
                              suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                              **kwargs):
    """TODO: Docstring
    """
    cov_table = tables.gen_table_metric_all(data_df, suites, f'{level}_coverage')
    cov_late_table = tables.gen_table_metric_all(data_df, suites, f'{level}_untimely_coverage')
    overpred_table = tables.gen_table_metric_all(data_df, suites, f'{level}_overpredictions')
    plot_overpredictions(cov_table, cov_late_table, overpred_table, **kwargs)


def plot_everything_overpredictions(data_df: Dict[str, pd.DataFrame],
                                    level: str = 'L2C',
                                    suites: List[Tuple[str, str]] = [('spec06', 'one_phase')],
                                    add_mean: bool = True,
                                    **kwargs):
    """TODO: Docstring
    """
    for suite, phase in suites:
        print(f'=== {suite} {phase} ===')
        cov_table = tables.gen_table_metric(data_df, suite, phase, f'{level}_coverage', add_mean=add_mean).T
        cov_table.columns = [utils.clean_trace_name(tr) for tr in cov_table.columns]

        cov_late_table = tables.gen_table_metric(data_df, suite, phase, f'{level}_untimely_coverage', add_mean=add_mean).T
        cov_late_table.columns = [utils.clean_trace_name(tr) for tr in cov_late_table.columns]

        overpred_table = tables.gen_table_metric(data_df, suite, phase, f'{level}_overpredictions', add_mean=add_mean).T
        overpred_table.columns = [utils.clean_trace_name(tr) for tr in overpred_table.columns]

        plot_overpredictions(cov_table, cov_late_table, overpred_table, **kwargs)
'''
