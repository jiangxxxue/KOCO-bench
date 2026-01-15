import os.path
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab


class DUST3R_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'dust3r'
        baseline_folder = 'dust3r'
        default_parameters = {'verbose': 1, 'max_rgb': 10}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def is_installed(self):
        return os.path.isdir(self.baseline_path)

    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/extra-files/dust3r/dust3r_execute.py")


