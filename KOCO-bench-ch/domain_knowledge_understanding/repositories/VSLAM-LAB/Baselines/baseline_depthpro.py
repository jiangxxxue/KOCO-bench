import os.path
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab


class DEPTHPRO_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'depthpro'
        baseline_folder = 'ml-depth-pro'

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder)

        self.default_parameters = {'verbose': 1, 'max-depth': 8, 'min-depth': 0.5, 'anchor': 0}

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def is_installed(self):
        return os.path.isdir(os.path.join(self.baseline_path, 'checkpoints'))

    def info_print(self):
        super().info_print()
        print(f"Default executable: extra-files/depth_estimation/depthpro_vslamlab.py")
