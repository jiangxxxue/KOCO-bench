import os
import yaml
import shutil
from huggingface_hub import hf_hub_download
from zipfile import ZipFile

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile
from utilities import decompressFile

from Evaluate.align_trajectories import align_trajectory_with_groundtruth
from Evaluate import metrics


class HAMLYN_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('hamlyn', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

    def download_sequence_data(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)

        # Variables
        compressed_name = sequence_name
        compressed_name_ext = compressed_name + '.zip'
        repo_id = self.url_download_root

        # Download the compressed file
        file_path = hf_hub_download(repo_id=repo_id, filename=compressed_name_ext, repo_type='dataset')
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_path)

        # Download instrinsics file
        intrinsics_file = f"intrinsics_{sequence_name}.txt"
        file_path = hf_hub_download(repo_id=repo_id, filename=intrinsics_file, repo_type='dataset')
        intrinsics_txt = os.path.join(sequence_path, intrinsics_file)
        with open(file_path, 'rb') as f_src, open(intrinsics_txt, 'wb') as f_dest:
            f_dest.write(f_src.read())

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        image01_path = os.path.join(sequence_path, 'image01')
        if os.path.exists(image01_path):
            os.rename(image01_path, rgb_path)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                name, ext = os.path.splitext(filename)
                ts = float(name) / self.rgb_hz
                file.write(f"{ts:.5f} rgb/{filename}\n")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_file = os.path.join(sequence_path, f"intrinsics_{sequence_name}.txt")
        with open(calibration_file, 'r') as file:
            lines = file.readlines()

        fx, _, cx, _ = map(float, lines[0].split())
        _, fy, cy, _ = map(float, lines[1].split())
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        return

    def remove_unused_files(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        calibration_file = os.path.join(sequence_path, f"intrinsics_{sequence_name}.txt")
        if os.path.exists(calibration_file):
            os.remove(calibration_file)

        depth01_folder = os.path.join(sequence_path, 'depth01')
        if os.path.exists(depth01_folder):
            shutil.rmtree(depth01_folder)

        depth02_folder = os.path.join(sequence_path, 'depth02')
        if os.path.exists(depth02_folder):
            shutil.rmtree(depth02_folder)

        image02_folder = os.path.join(sequence_path, 'image02')
        if os.path.exists(image02_folder):
            shutil.rmtree(image02_folder)