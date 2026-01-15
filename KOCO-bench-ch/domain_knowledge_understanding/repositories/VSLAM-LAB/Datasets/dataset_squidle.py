import os, yaml
import pandas as pd
import numpy as np

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile
from Datasets.dataset_utilities import undistort_rgb_rad_tan, undistort_depth_rad_tan

class SQUIDLE_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('squidle', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = self.sequence_names
