import os
import yaml
import shutil

import glob
import numpy as np
from scipy.spatial.transform import Rotation as R

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics


class SEVENSCENES_dataset(DatasetVSLAMLab):
    scenes = ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']

    def __init__(self, benchmark_path):

        # Initialize the dataset
        super().__init__("7scenes", benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_seq-', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):

        # Variables
        sequence_group = self.find_sequence_group(sequence_name)
        compressed_name = sequence_group
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = compressed_name
        download_url = os.path.join(self.url_download_root, compressed_name_ext)

        # Constants
        compressed_file = os.path.join(self.dataset_path, compressed_name_ext)
        decompressed_folder = os.path.join(self.dataset_path, decompressed_name)

        # Download the compressed file
        if not os.path.exists(compressed_file):
            downloadFile(download_url, self.dataset_path)

        # Decompress the file
        if not os.path.exists(decompressed_folder):
            decompressFile(compressed_file, self.dataset_path)

        # Variables
        compressed_name = sequence_name.replace(sequence_group + '_', '')
        compressed_name_ext = compressed_name + '.zip'
        decompressed_name = compressed_name

        # Constants
        compressed_file = os.path.join(self.dataset_path, sequence_group, compressed_name_ext)
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        decompressed_folder = sequence_path

        if not os.path.exists(decompressed_folder):
            decompressFile(compressed_file, self.dataset_path)
            os.rename(os.path.join(self.dataset_path, decompressed_name), sequence_path)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')

        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path, exist_ok=True)
            rgb_files = glob.glob(os.path.join(sequence_path, '*.color.png'))
            for png_path in rgb_files:
                rgb_name = os.path.basename(png_path)
                rgb_name = rgb_name.replace("frame-", "")
                rgb_name = rgb_name.replace("color.", "")
                shutil.copy(png_path, os.path.join(rgb_path, rgb_name))

            png_files = glob.glob(os.path.join(sequence_path, '*.png'))
            for png_file in png_files:
                os.remove(png_file)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        png_files = [file for file in os.listdir(rgb_path) if file.endswith('.png')]
        png_files.sort()
        with open(rgb_txt, 'w') as file:
            for iPNG, png_file in enumerate(png_files, start=0):
                ts = iPNG / self.rgb_hz
                file.write(f"{ts:.5f} rgb/{png_file}\n")

    def create_calibration_yaml(self, sequence_name):

        fx, fy, cx, cy = 585.0, 585.0, 320.0, 240.0
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
        pose_files = glob.glob(os.path.join(sequence_path, '*.pose.txt'))
        pose_files.sort()
        with open(groundtruth_txt, 'w') as destination_file:
            for iRGB, gt0 in enumerate(pose_files, start=0):
                with open(gt0, 'r') as source_file:
                    T = []
                    for line in source_file:
                        row = [float(x) for x in line.split()]
                        T.append(row)
                    tx = T[0][3]
                    ty = T[1][3]
                    tz = T[2][3]
                    rotation_matrix = np.array([[T[0][0], T[0][1], T[0][2]],
                                                [T[1][0], T[1][1], T[1][2]],
                                                [T[2][0], T[2][1], T[2][2]]])
                    rotation = R.from_matrix(rotation_matrix)
                    quaternion = rotation.as_quat()
                    qx = quaternion[0]
                    qy = quaternion[1]
                    qz = quaternion[2]
                    qw = quaternion[3]
                    ts = iRGB / self.rgb_hz
                    line2 = "{:.5f} {} {} {} {} {} {} {}\n".format(ts, tx, ty, tz, qx, qy, qz, qw)
                    destination_file.write(line2)
                os.remove(gt0)

    def remove_unused_files(self, sequence_name):
        sequence_group = self.find_sequence_group(sequence_name)
        sequences_folder = os.path.join(self.dataset_path, sequence_group)

        if os.path.exists(sequences_folder):
            if os.path.exists(os.path.join(sequences_folder, sequence_group + '.png')):
                os.remove(os.path.join(sequences_folder, sequence_group + '.png'))
            if os.path.exists(os.path.join(sequences_folder, 'TestSplit.txt')):
                os.remove(os.path.join(sequences_folder, 'TestSplit.txt'))
            if os.path.exists(os.path.join(sequences_folder, 'TrainSplit.txt')):
                os.remove(os.path.join(sequences_folder, 'TrainSplit.txt'))

            zipFile = sequence_name.replace(sequence_group + '_', '') + '.zip'
            if os.path.exists(os.path.join(sequences_folder, zipFile)):
                os.remove(os.path.join(sequences_folder, zipFile))

            if not os.listdir(sequences_folder):
                shutil.rmtree(sequences_folder)

    def find_sequence_group(self, sequence_name):
        for scene in self.scenes:
            if scene in sequence_name:
                return scene