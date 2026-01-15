import os
import yaml
import shutil

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile
from path_constants import VSLAMLAB_BENCHMARK

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics


class LAMAR_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('lamar', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):

        # Variables
        compressed_name = sequence_name
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = sequence_name
        download_url = self.url_download_root

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, VSLAMLAB_BENCHMARK)

        # Decompress the file
        if os.path.exists(self.dataset_path):
            if not os.listdir(self.dataset_path):
                shutil.rmtree(self.dataset_path)

        if not os.path.exists(decompressed_folder):
            decompressFile(compressed_file, self.dataset_path)

    def create_rgb_folder(self, sequence_name):
        return

    def create_rgb_txt(self, sequence_name):
        return

    def create_calibration_yaml(self, sequence_name):
        return

    def create_groundtruth_txt(self, sequence_name):
        return

    def remove_unused_files(self, sequence_name):
        return

    def evaluate_trajectory_accuracy(self, trajectory_txt, groundtruth_txt):
        return
