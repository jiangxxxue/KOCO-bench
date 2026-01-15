import os, yaml
import numpy as np
import subprocess
import shutil
from PIL import Image
from tqdm import tqdm
from Datasets.DatasetVSLAMLab import DatasetVSLAMLab
from utilities import downloadFile, decompressFile

class SWEETCORALS_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path):
        # Initialize the dataset
        super().__init__('sweetcorals', benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Get download url
        self.url_download_root = data['url_download_root']

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Get resolution size
        self.resolution_size = data['resolution_size']


    # def download_sequence_data(self, sequence_name):
    #     if(os.path.isdir(self.dataset_path) and not os.listdir(self.dataset_path)):
    #         shutil.rmtree(self.dataset_path)
    #         subprocess.run(["git", "clone", self.url_download_root, self.dataset_path], check=True)
    #         subprocess.run(["git", "lfs", "pull"], check=True, cwd=self.dataset_path)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        corrected_path = os.path.join(sequence_path, 'corrected', 'images')

        if not os.path.exists(rgb_path) and os.path.exists(corrected_path):
            os.makedirs(rgb_path, exist_ok=True)   
            estimate_new_resolution = True
            for file in tqdm(os.listdir(corrected_path), desc="    resizing images"):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    with Image.open(os.path.join(corrected_path, file)) as img:
                        img.load()
                        if estimate_new_resolution:
                            scaled_height = np.sqrt(self.resolution_size[0] * self.resolution_size[1] * img.size[1] / img.size[0])
                            scaled_width = self.resolution_size[0] * self.resolution_size[1] / scaled_height
                            scaled_height = int(scaled_height)
                            scaled_width = int(scaled_width)
                            estimate_new_resolution = False
                    resized_img = img.resize((scaled_width, scaled_height), Image.LANCZOS)
                    resized_img.save(os.path.join(rgb_path, file))

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for i, filename in enumerate(rgb_files):
                _, _ = os.path.splitext(filename)
                ts = i / self.rgb_hz
                file.write(f"{ts:.5f} rgb/{filename}\n")
        
    def create_calibration_yaml(self, sequence_name):
        fx, fy, cx, cy = 363.17280905365755 ,363.17280905365755 ,296 ,259 # Got with COLMAP
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        self.write_calibration_yaml('PINHOLE', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    # def create_groundtruth_txt(self, sequence_name):
    #     sequence_path = os.path.join(self.dataset_path, sequence_name)
    #     groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')
    #     groundtruth_csv = os.path.join(sequence_path, 'groundtruth.csv')
    #     groundtruth_raw_txt = os.path.join(sequence_path, 'groundtruth_raw.txt')
        
    #     with open(groundtruth_raw_txt, 'r') as file:
    #         lines = file.readlines()

    #     number_of_grountruth_header_lines = 1
    #     new_lines = lines[number_of_grountruth_header_lines:]
    #     with open(groundtruth_txt, 'w') as file:
    #         file.writelines(new_lines)

    #     header = "ts,tx,ty,tz,qx,qy,qz,qw\n"
    #     new_lines.insert(0, header)
    #     with open(groundtruth_csv, 'w') as file:
    #         for line in new_lines:
    #             values = line.split()
    #             file.write(','.join(values) + '\n')
        
    # def remove_unused_files(self, sequence_name):
    #     sequence_path = os.path.join(self.dataset_path, sequence_name)
    #     os.remove(os.path.join(sequence_path, 'calibration.txt'))