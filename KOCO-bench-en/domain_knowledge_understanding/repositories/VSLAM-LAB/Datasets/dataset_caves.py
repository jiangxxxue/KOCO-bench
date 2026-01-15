import os
import yaml
import shutil
import csv

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile
from path_constants import VSLAMLAB_BENCHMARK

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics


class CAVES_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('caves', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']
        self.url_download_timestamps = data['url_download_timestamps']
        self.url_download_groundtruth = data['url_download_groundtruth']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):
        # Variables
        compressed_name = "undistorted_frames"
        compressed_name_ext = compressed_name + '.zip'
        download_url = self.url_download_root
        timestamps_url = self.url_download_timestamps
        groundtruth_url = self.url_download_groundtruth
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)
            os.rename(os.path.join(self.dataset_path, 'undistorted_frames.zip?download=1'), os.path.join(self.dataset_path, compressed_name_ext))

        # Download timestamps file
        downloadFile(timestamps_url, self.dataset_path)
        os.rename(os.path.join(self.dataset_path, 'undistorted_frames_timestamps.txt?download=1'), os.path.join(self.dataset_path, 'undistorted_frames_timestamps.txt'))
        decompressFile(compressed_file, sequence_path)
        os.rename(os.path.join(sequence_path, 'undistorted_frames'), os.path.join(sequence_path, 'rgb'))
        
        # Download full dataset to get the odometry file
        downloadFile(groundtruth_url, self.dataset_path)
        os.rename(os.path.join(self.dataset_path, 'full_dataset.zip?download=1'), os.path.join(self.dataset_path, 'full_dataset.zip'))
        decompressFile(os.path.join(self.dataset_path, 'full_dataset.zip'), self.dataset_path)

    def create_rgb_folder(self, sequence_name):
        return

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        timestamps_txt = os.path.join(self.dataset_path, 'undistorted_frames_timestamps.txt')

        with open(timestamps_txt, 'r') as ts_file:
            with open(rgb_txt, 'w') as file:
                for line in ts_file:
                    name, ts = line.strip().split('\t')
                    file.write(f"{ts} rgb/{name}\n") 

    def create_calibration_yaml(self, sequence_name):
        fx, fy, cx, cy = 405.6384738851233, 405.588335378204, 189.9054317917407, 139.9149578253755
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')

        groundtruth_txt_0 = os.path.join(self.dataset_path, 'full_dataset', 'odometry.txt')

        columns = [
            '%time',
            'field.pose.pose.position.x',
            'field.pose.pose.position.y',
            'field.pose.pose.position.z',
            'field.pose.pose.orientation.x',
            'field.pose.pose.orientation.y',
            'field.pose.pose.orientation.z',
            'field.pose.pose.orientation.w'
        ]

        with open(groundtruth_txt_0, 'r') as infile:
            reader = csv.DictReader(infile)
            
            with open(groundtruth_txt, 'w') as outfile:
                writer = csv.writer(outfile, delimiter=' ')

                for row in reader:
                    row_data = [
                        float(row[columns[0]]) / 1e9
                    ] + [row[column] for column in columns[1:]]
                    
                    writer.writerow(row_data)

    def remove_unused_files(self, sequence_name):
        timestamps_txt = os.path.join(self.dataset_path, 'undistorted_frames_timestamps.txt')
        full_dataset_folder = os.path.join(self.dataset_path, 'full_dataset')
        full_dataset_zip = os.path.join(self.dataset_path, 'full_dataset.zip')

        if os.path.exists(timestamps_txt):
            os.remove(timestamps_txt)
        
        if os.path.exists(full_dataset_folder):
            shutil.rmtree(full_dataset_folder)
        
        if os.path.exists(full_dataset_zip):
            os.remove(full_dataset_zip)
