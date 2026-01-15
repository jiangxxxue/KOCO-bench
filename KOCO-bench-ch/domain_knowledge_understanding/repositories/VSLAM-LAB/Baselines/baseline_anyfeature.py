import os.path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

from path_constants import VSLAMLAB_BASELINES


class ANYFEATURE_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'anyfeature'
        baseline_folder = 'AnyFeature-VSLAM'
        baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        default_parameters = ['Vis:1', 'Feat:orb32', f'anyfeat:{baseline_path}/']

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder)