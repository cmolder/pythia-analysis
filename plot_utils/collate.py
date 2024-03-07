import os
from typing import Dict, List, Set, Optional

import pandas as pd

from naboo_utils import file, table
class StudyCollator():
    """"Helper class that gathers and organizes studies.
    """
    def __init__(self, sim_dir: str, study_name: str,
                 mix_file: str, benchmark_file: str, suite_file: str, mix_set: str,
                 baseline_study: str      = "stu_test",
                 baseline_experiment: str = "_baseline",
                 num_threads: int         =  16,
                 experiments: Optional[List[str]] = None,
                 suites: Optional[Set[str]] = None):
        self.study_path = os.path.join(sim_dir, study_name)
        self.baseline_study_path = os.path.join(sim_dir, baseline_study)

        if experiments is not None and baseline_experiment not in experiments:
            experiments.append(baseline_experiment)

        self.study = file.ChampsimStudy.FromStudyDir(self.study_path, experiments=experiments)
        self.baseline_study = file.ChampsimStudy.FromStudyDir(self.baseline_study_path)
        self.baseline_experiment = baseline_experiment

        # Process the study
        print(f"~~~ {study_name} ~~~")
        self.study.read_all(num_threads=num_threads,
                            baseline_study=self.baseline_study,
                            baseline_exp=self.baseline_experiment)

        # Create a tabler and precompute benchmark/suite statistics.
        self.tabler = table.ChampsimTabler(self.study, suite_file, benchmark_file, mix_file, mix_set)
        print()

    def __getitem__(self, item):
        assert isinstance(item, str), "StudyCollator must be indexed by experiment name"
        return self.get_experiment(item)
    
    def __str__(self):
        return f"StudyCollator({self.study_path})"
    
    def __repr__(self):
        return str(self)

    def get_experiment(self, experiment: str):
        return ExperimentCollator(self, experiment)


class ExperimentCollator():
    """Helper class that gathers and organizes experiment statistics.
    """
    def __init__(self, study: StudyCollator, experiment_name: str):
        self.study = study
        self.experiment_name = experiment_name

    def __str__(self):
        return f"ExperimentCollator({self.study.study_path}, {self.experiment_name})"

    def __repr__(self):
        return str(self)

    def get_suite_statistic(self, stat: str, 
                            suites: Optional[List[str]] = None, 
                            use_weights: bool = True):
        return self.study.tabler.table_statistic_suite(
            stat,
            experiments=[self.experiment_name],
            suites=suites,
            use_weights=use_weights,
        )

    def get_benchmark_statistic(self, suite: str, stat: str,
                                use_weights: bool = True):
        benchmarks = self.study.tabler.get_suite_benchmarks(suite)
        if benchmarks is not None:
            benchmarks = sorted(list(benchmarks))

        return self.study.tabler.table_statistic_benchmark(
            stat,
            experiments=[self.experiment_name],
            benchmarks=benchmarks,
            use_weights=use_weights
        )
    
    def get_mix_statistic(self, suite: str, stat: str):
        mixes = self.study.tabler.get_suite_mixes(suite)
        if mixes is not None:
            mixes = sorted(list(mixes))

        return self.study.tabler.table_statistic_mix(
            stat,
            experiments=[self.experiment_name],
            mixes=mixes
        )

def get_suite_statistic(experiments: Dict[str, ExperimentCollator], 
                        stat: str,
                        suites: Optional[List[str]] = None,
                        use_weights: bool = True):
    rows = {}
    for exp_name, exp in experiments.items():
        rows[exp_name] = exp.get_suite_statistic(
            stat, suites=suites, use_weights=use_weights
        ).iloc[0]
    return pd.DataFrame(rows)

def get_benchmark_statistic(experiments: Dict[str, ExperimentCollator], 
                            suite: str, 
                            stat: str,
                            use_weights: bool = True):
    rows = {}
    for exp_name, exp in experiments.items():
        rows[exp_name] = exp.get_benchmark_statistic(
            suite, stat, 
            use_weights=use_weights, 
        ).iloc[0]
    return pd.DataFrame(rows)

def get_mix_statistic(experiments: Dict[str, ExperimentCollator], 
                      suite: str, 
                      stat: str):
    rows = {}
    for exp_name, exp in experiments.items():
        rows[exp_name] = exp.get_mix_statistic(
            suite, stat
        ).iloc[0]
    return pd.DataFrame(rows)