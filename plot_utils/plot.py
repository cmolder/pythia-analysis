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
                        suite: str,
                        stat: str,
                        benchmarks: Optional[List[str]] = None,
                        add_mean: bool = True,
                        use_weights: bool = True) -> pd.DataFrame:
    """Get the table for benchmarks, with some extra processing
    for plotting.
    """
    table = collate.get_benchmark_statistic(
        experiments, suite, stat,
        use_weights=use_weights).T
    if benchmarks is not None:
        table = table[benchmarks]
    if add_mean:
        # TODO: Use naboo tabler to get mean
        mean = get_mean(stat, table)
        table[get_mean_string(stat)] = mean
    return table

def get_suite_table(experiments: Dict[str, collate.ExperimentCollator],
                    stat: str,
                    suites: Optional[List[str]] = None,
                    add_mean: bool = True,
                    use_weights: bool = True) -> pd.DataFrame:
    """Get the table for a suite, with some extra processing
    for plotting.
    """
    table = collate.get_suite_statistic(
        experiments, stat,
        suites=suites,
        use_weights=use_weights).T
    if add_mean:
        # TODO: Use naboo tabler to get mean
        mean = get_mean(stat, table)
        table[get_mean_string(stat)] = mean
    return table

def get_mix_table(experiments: Dict[str, collate.ExperimentCollator],
                  suite: str,
                  stat: str,
                  mixes: Optional[List[str]] = None) -> pd.DataFrame:
    """Get the table for mixes, with some extra processing 
    for plotting."""
    table = collate.get_mix_statistic(
        experiments, suite, stat).T
    if mixes is not None:
        table = table[mixes]
    return table

def plot_table(table: pd.DataFrame,
               secondary_tables: Optional[List[pd.DataFrame]] = None,
               # Plotting kwargs
               figsize: Tuple[int, int] = None,
               dpi: int = None,
               legend: bool = True,
               add_suptitle: bool = True, suptitle: str = "",
               add_xlabel: bool = True, xlabel: str = "Benchmark",
               add_ylabel: bool = True, ylabel: str = "",
               colors: defaultdict = defaultdict(lambda: None),
               secondary_colors: Optional[List[str]] = ["lightgray"],
               secondary_hatches: Optional[List[str]] = [None],
               secondary_labels: Optional[List[str]] = None,
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
    
    Xs, ys, ys_secondary = {}, {}, []
    min_y, max_y = 0.0, 0.0
    num_entries, num_traces = len(table), len(table.columns)
    gap, margin = 2, 2

    # Plot
    for i, (index, row) in enumerate(table.iterrows()):
        Xs[index] = (np.arange(num_traces) * (num_entries + gap)) + i
        ys[index] = row.values
        min_y = min(min_y, min(row.values))
        max_y = max(max_y, max(row.values))
    if secondary_tables is not None:
        for tab in secondary_tables:
            ys_secondary_tab = {}
            for i, (index, row) in enumerate(tab.iterrows()):
                ys_secondary_tab[index] = row.values
                min_y = min(min_y, min(row.values))
                max_y = max(max_y, max(row.values))
            ys_secondary.append(ys_secondary_tab)
            

    # Bars
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for x_i, (k, x) in enumerate(Xs.items()):
        for sec_i, dat_secondary in enumerate(ys_secondary):
            if k in dat_secondary:
                # Plot sceondary bar from bottom
                ax.bar(x, dat_secondary[k], 
                       color=secondary_colors[sec_i], 
                       hatch=secondary_hatches[sec_i],
                       width=1.0, #edgecolor='gray', 
                       edgecolor='black',
                       linewidth=0.3, 
                       zorder=1 - (sec_i * 0.01),
                       label=(secondary_labels[sec_i] if secondary_labels is not None and x_i == 0 else None))
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
    if add_suptitle and suptitle != "":
        fig.suptitle(suptitle, fontsize=9)
    
    fig.tight_layout()

"""
Plotting functions
"""
def plot_benchmark_statistic(experiments: Dict[str, collate.ExperimentCollator],
                             suite_name: str,
                             statistic_name: str,
                             secondary_statistic_names: Optional[List[str]] = None,
                             benchmarks: Optional[List[str]] = None,
                             add_mean: bool = True,
                             **kwargs):
    """TODO: Docstring
    """
    # Gather table(s)
    table = get_benchmark_table(experiments, suite_name, statistic_name,
                                benchmarks=benchmarks, add_mean=add_mean)
    
    secondary_tables = None
    if secondary_statistic_names is not None:
        secondary_tables = []
        for stat in secondary_statistic_names:
            secondary_tables.append(get_benchmark_table(
                experiments, suite_name, stat,
                benchmarks=benchmarks,
                add_mean=add_mean))

    plot_table(table, secondary_tables=secondary_tables, **kwargs)

def plot_suite_statistic(experiments: Dict[str, collate.ExperimentCollator],
                         statistic_name: str,
                         secondary_statistic_names: Optional[List[str]] = None,
                         suites: Optional[List[str]] = None,
                         add_mean: bool = True,
                         **kwargs):
    """TODO: Docstring
    """
    table = get_suite_table(experiments, statistic_name,
                            suites=suites, add_mean=add_mean)
    
    secondary_tables = None
    if secondary_statistic_names is not None:
        secondary_tables = []
        for stat in secondary_statistic_names:
            secondary_tables.append(get_suite_table(
                experiments, stat,
                suites=suites,
                add_mean=add_mean))

    plot_table(table, secondary_tables=secondary_tables, **kwargs)

def plot_benchmark_overpredictions(experiments: Dict[str, collate.ExperimentCollator],
                                   suite_name: str,
                                   cache_name: Optional[str] = "L2C",
                                   benchmarks: Optional[List[str]] = None,
                                   add_mean: bool = True,
                                   **kwargs):
    cov_table = get_benchmark_table(
        experiments, suite_name, 
        f"cache.{cache_name}.amean.prefetch_coverage",
        benchmarks=benchmarks,
        add_mean=add_mean
    )
    cov_untimely_table = get_benchmark_table(
        experiments, suite_name, 
        f"cache.{cache_name}.amean.prefetch_coverage_untimely",
        benchmarks=benchmarks,
        add_mean=add_mean
    )
    overpred_table = get_benchmark_table(
        experiments, suite_name, 
        f"cache.{cache_name}.amean.prefetch_overpredictions",
        benchmarks=benchmarks,
        add_mean=add_mean
    )

    # Uncovered table: always equals one
    uncov_table = cov_table.copy()
    uncov_table[:] = 100.0

    overpred_table[:] += 100.0

    plot_table(
        cov_table, 
        secondary_tables=[cov_untimely_table, uncov_table, overpred_table], 
        secondary_colors=["lightgray", "darkgray", "white"],
        secondary_hatches=[None, None, "...."],
        secondary_labels=["(covered, late)", "(not covered)", "(useless)"],
        ylabel="Demand misses (%)",
        **kwargs
    )

def plot_suite_overpredictions(experiments: Dict[str, collate.ExperimentCollator],
                               cache_name: Optional[str] = "L2C",
                               suites: Optional[List[str]] = None,
                               add_mean: bool = True,
                               **kwargs):

    cov_table = get_suite_table(
        experiments, 
        f"cache.{cache_name}.amean.prefetch_coverage",
        suites=suites,
        add_mean=add_mean
    )
    cov_untimely_table = get_suite_table(
        experiments, 
        f"cache.{cache_name}.amean.prefetch_coverage_untimely",
        suites=suites,
        add_mean=add_mean
    )
    overpred_table = get_suite_table(
        experiments, 
        f"cache.{cache_name}.amean.prefetch_overpredictions",
        suites=suites,
        add_mean=add_mean
    )

    # Uncovered table: always equals one
    uncov_table = cov_table.copy()
    uncov_table[:] = 100.0

    overpred_table[:] += 100.0

    plot_table(
        cov_table, 
        secondary_tables=[cov_untimely_table, uncov_table, overpred_table], 
        secondary_colors=["lightgray", "darkgray", "white"],
        secondary_hatches=[None, None, "...."],
        secondary_labels=["(covered, late)", "(not covered)", "(useless prefetches)"],
        ylabel="Demand misses (%)",
        **kwargs
    )