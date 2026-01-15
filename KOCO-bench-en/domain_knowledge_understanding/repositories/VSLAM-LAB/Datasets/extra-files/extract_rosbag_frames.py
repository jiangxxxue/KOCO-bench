import rosbag
from cv_bridge import CvBridge
import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--rosbag_path', type=str, help=f"rosbag path")
    parser.add_argument('--sequence_path', type=str, help=f"sequence_path")
    parser.add_argument('--image_topic', type=str, help=f"image topic")

    args = parser.parse_args()

    bridge = CvBridge()
    rosbag_path = args.rosbag_path
    image_topic = args.image_topic
    sequence_path = args.sequence_path
    rgb_path = os.path.join(sequence_path, 'rgb')
    rgb_txt = os.path.join(sequence_path, 'rgb.txt')

    with rosbag.Bag(rosbag_path, 'r') as bag:
        with open(rgb_txt, 'w') as file:
            for topic, msg, t in tqdm(bag.read_messages(topics=[image_topic]), desc=f'Extracting frames from {image_topic} ...'):
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                except Exception as e:
                    print(f"Could not convert image: {e}")
                    continue

                image_name = f"{t}.png"
                image_path = os.path.join(rgb_path,image_name)
         
                cv2.imwrite(image_path, cv_image)
                file.write(f"{(float(str(t))*10e-10):.5f} rgb/{image_name}\n")

if __name__ == "__main__":
    main()

  
