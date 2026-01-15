import cv2
import subprocess
import numpy as np
from tqdm import tqdm
import os, yaml, shutil

from Datasets.DatasetVSLAMLab import DatasetVSLAMLab

class ARIEL_dataset(DatasetVSLAMLab):
    def __init__(self, benchmark_path, dataset_name = 'ariel'):
        # Initialize the dataset
        super().__init__(dataset_name, benchmark_path)

        # Load settings from .yaml file
        with open(self.yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        # Create sequence_nicknames
        self.sequence_nicknames = [s.replace('_', ' ') for s in self.sequence_names]

        # Rosbag image topic
        self.image_topic = data['image_topic']

        # Camera calibration
        self.calibration = data['calibration']

        # Camera calibration configurations
        self.calibration_configs = data['calibration_configurations']

        # Ground-Truth file names
        self.gt_file_names = data['gt_files']

        # Rosbag file names
        self.rosbag_names = data['rosbag_names']

    def download_sequence_data(self, sequence_name):
        rosbag_names = self.get_rosbag_names(sequence_name)
        rosbag_path = os.path.join(self.dataset_path, f"{rosbag_names[0]}.bag")
        gt_file_name = self.gt_file_names[self.seq_name_wo_cam(sequence_name)]
        gt_file = os.path.join(self.dataset_path, gt_file_name)

        if (not os.path.exists(rosbag_path)) or (not os.path.exists(gt_file)):
            message = (
                f"[dataset_{self.dataset_name}.py] \n    The \'{self.dataset_name}\' dataset cannot be downloaded automatically. "
                f"\n\n    Please manually download the rosbags and ground-truth files from: \'https://github.com/ntnu-arl/underwater-datasets\'"
                f"\n    and place them in: \'{self.dataset_path}\'. "
                f"\n\n    ros-bags associated with sequence '{sequence_name}' are: {rosbag_names}"
                f"\n\n    The ground-truth file associated with sequence '{sequence_name}' is: '{gt_file_name}'"
                f"\n\n    WHEN PROVIDING THE ROSBASGS NOTE THAT THE SEQUENCES ARE DIVIDED WITH TEMPORAL ORDER!!!")
            print(message)
            exit(0)

    def create_rgb_folder(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rosbag_names = self.get_rosbag_names(sequence_name)
        image_topic = self.get_rosbag_image_topic(sequence_name)

        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path, exist_ok=True)
            for rosbag_name in rosbag_names:
                rosbag_path = os.path.join(self.dataset_path, f"{rosbag_name}.bag")
                if not os.path.exists(rosbag_path):
                    break
                command = f"pixi run -e ros-env extract-rosbag-frames {rosbag_path} {rgb_path} {image_topic}"
                subprocess.run(command, shell=True)

    def create_rgb_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        rgb_files.sort()
        with open(rgb_txt, 'w') as file:
            for iRGB, filename in enumerate(rgb_files, start=0):
                name, ext = os.path.splitext(filename)
                ts = float(name) / 10e8
                file.write(f"{ts:.5f} rgb/{filename}\n")

    def create_calibration_yaml(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        rgb_path = os.path.join(sequence_path, 'rgb')

        rgb_files = [f for f in os.listdir(rgb_path) if os.path.isfile(os.path.join(rgb_path, f))]
        image_0 = cv2.imread(os.path.join(rgb_path, rgb_files[0]))
        h, w, channels = image_0.shape
        K, D = self.get_camera_calibration(sequence_name)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)

        print(f"[dataset_{self.dataset_name}.py] Undistorting rgb images with OpenCV fisheye model ...")
        for rgb_file in tqdm(rgb_files):
            image_path = os.path.join(rgb_path, rgb_file)
            image = cv2.imread(image_path)
            undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(image_path, undistorted_img)

        fx, fy, cx, cy = new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2]
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0
        self.write_calibration_yaml('OPENCV', fx, fy, cx, cy, k1, k2, p1, p2, k3, sequence_name)

    def create_groundtruth_txt(self, sequence_name):
        sequence_path = os.path.join(self.dataset_path, sequence_name)
        groundtruth_txt = os.path.join(sequence_path, 'groundtruth.txt')

        gt_file_name = self.gt_file_names[self.seq_name_wo_cam(sequence_name)]
        gt_file = os.path.join(self.dataset_path,gt_file_name)
        shutil.copy(gt_file, groundtruth_txt)

    def remove_unused_files(self, sequence_name):
        return


    def get_camera_calibration(self, sequence_name):
        cam = self.get_camera(sequence_name)
        cal_config = self.get_calibration_configs(sequence_name)

        switch = {
            'Traj_3_id1': lambda: self.calibration[cam][cal_config],
            'Traj_4_id2': lambda: self.calibration[cam][cal_config]
        }

        cal = switch[self.seq_name_wo_cam(sequence_name)]()
        K = np.array([[cal[0], 0, cal[2]], [0, cal[1], cal[3]], [0, 0, 1]])
        D = np.array([cal[4], cal[5], cal[6], cal[7]])
        return K, D


    def seq_name_wo_cam(self, sequence_name):
        return sequence_name.replace(f"-{self.get_camera(sequence_name)}", "")

    def get_rosbag_image_topic(self, sequence_name):
        return f"{self.image_topic}/{self.get_camera(sequence_name)}"

    def get_rosbag_names(self, sequence_name):
        return self.rosbag_names[self.seq_name_wo_cam(sequence_name)]

    def get_camera(self, sequence_name):
        if 'cam0' in sequence_name:
            return 'cam0'
        if 'cam1' in sequence_name:
            return 'cam1'
        if 'cam3' in sequence_name:
            return 'cam2'
        if 'cam4' in sequence_name:
            return 'cam3'

        raise ValueError(f"Invalid camera (cam0, cam1, cam3, cam4): '{sequence_name}'")

    def get_calibration_configs(self, sequence_name):
        return self.calibration_configs[self.seq_name_wo_cam(sequence_name)]
    
    def get_download_issues(self, sequence_names):
        
        missing_rosbags = []
        for sequence_name in sequence_names: 
            sequence_path = os.path.join(self.dataset_path, sequence_name)
            rosbag_names = self.get_rosbag_names(sequence_name)
            for rosbag_name in rosbag_names:
                if not os.path.isfile(os.path.join(self.dataset_path, rosbag_name)):
                    missing_rosbags.append(os.path.join(self.dataset_path, rosbag_name))
        
        if not missing_rosbags:
            return []
        
        solution_msg = f'Please manually download the rosbags and ground-truth files from: \'https://github.com/ntnu-arl/underwater-datasets\', '
        solution_msg += f'\n         and place them in: \'{self.dataset_path}\'. '
        solution_msg += f'Missing rosbags are:'
        for missing_rosbag in missing_rosbags:
            solution_msg += f'\n             {missing_rosbag}.bag'

        issues = []
        issue = {}
        issue['name'] = 'Manual download'
        issue['description'] = f"The \'{self.dataset_name}\' dataset cannot be downloaded automatically."
        issue['solution'] = solution_msg
        issue['mode'] = '\033[92mmanual download\033[0m'
        issues.append(issue)
        return issues
