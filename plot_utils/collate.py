import os
from typing import Dict, List, Set, Optional

import pandas as pd

from naboo_utils import file, table
class StudyCollator():
    """"Helper class that gathers and organizes studies.
    """
    def __init__(self, sim_dir: str, study_name: str,
                 baseline_study: str      = "stu_test",
                 baseline_experiment: str = "_baseline",
                 num_threads: int         =  16,
                 weights_path: str        = "weights.toml",
                 experiments: Optional[List[str]] = None,
                 suites: Optional[Set[str]] = None):
        self.study_path = os.path.join(sim_dir, study_name)
        self.baseline_study_path = os.path.join(sim_dir, baseline_study)

        if experiments is not None and baseline_experiment not in experiments:
            experiments.append(baseline_experiment)

        self.study = file.ChampsimStudy(self.study_path, experiments=experiments)
        self.baseline_study = file.ChampsimStudy(self.baseline_study_path)
        self.baseline_experiment = baseline_experiment

        # Process the study
        self.study.read_all(num_threads=num_threads,
                            baseline_study=self.baseline_study,
                            baseline_exp=self.baseline_experiment)

        # Create a tabler and precompute benchmark/suite statistics.
        self.tabler = table.ChampsimTabler(self.study, 
                                                 weights_path=weights_path)
        self.tabler.generate_benchmark_statistics(num_threads=num_threads)
        self.tabler.generate_suite_statistics(num_threads=num_threads, suites=suites)

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

    def get_suite_statistic(self, statistic_name: str):
        table = self.study.tabler.table_suite_statistic(statistic_name)
        return table.loc[self.experiment_name, :]

    def get_benchmark_statistic(self, suite_name: str, statistic_name: str):
        table = self.study.tabler.table_benchmark_statistic(statistic_name, suite_name)
        return table[self.experiment_name]
    

def get_suite_statistic(experiments: Dict[str, ExperimentCollator], statistic_name: str):
    rows = {}
    for exp_name, exp in experiments.items():
        rows[exp_name] = exp.get_suite_statistic(statistic_name)
    return pd.DataFrame(rows)

def get_benchmark_statistic(experiments: Dict[str, ExperimentCollator], suite_name: str, statistic_name: str):
    rows = {}
    for exp_name, exp in experiments.items():
        rows[exp_name] = exp.get_benchmark_statistic(suite_name, statistic_name)
    return pd.DataFrame(rows)