
import pandas as pd
import argparse
import os
import re
import shutil

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def get_idx(file_name):
    return int(re.search(r'(\d+)', file_name).group(1))
def get_idx_string(file_name):
    return re.search(r'(\d+)', file_name).group(1)

# Set up argument parser
parser = argparse.ArgumentParser(description="Clean iterations from failed experiments.")
parser.add_argument('--exp_path', type=str, required=True, help="Path to experiment directory.")

# Parse arguments
args = parser.parse_args()

items = os.listdir(args.exp_path)
datasets = [item for item in items if os.path.isdir(os.path.join(args.exp_path, item))]

for dataset in datasets:
    dataset_path = os.path.join(args.exp_path, dataset)
    items = os.listdir(dataset_path)
    sequences = [item for item in items if os.path.isdir(os.path.join(dataset_path, item))]
    for sequence in sequences:
        sequence_path = os.path.join(dataset_path, sequence)
        items = os.listdir(sequence_path)

        trajectories = [item for item in items if os.path.isfile(os.path.join(sequence_path, item)) and 'KeyFrameTrajectory.txt' in item]
        trajectories.sort()

        system_outputs = [item for item in items if os.path.isfile(os.path.join(sequence_path, item)) and 'system_output' in item]
        system_outputs.sort()

        idx_last_trajectory = get_idx(trajectories[-1])
        idx_last_system_output = get_idx(system_outputs[-1])
        if idx_last_trajectory != idx_last_system_output:
            print(f"\n{SCRIPT_LABEL}{RED}Corrupted sequence:{RESET} {sequence_path}")
            print(f"    Last trajectory idx: {idx_last_trajectory}")
            print(f"    Last system output idx: {idx_last_system_output}")

            # Remove system_outputs
            # Update log_ablation_parameters.csv
            # Update log_run_sequence_time.csv
            # Remove evaluation folders

            log_ablation_parameters_csv = os.path.join(sequence_path, 'log_ablation_parameters.csv')
            if os.path.exists(log_ablation_parameters_csv):
                log_ablation_parameters_df = pd.read_csv(log_ablation_parameters_csv)

            log_run_sequence_time_csv = os.path.join(sequence_path, 'log_run_sequence_time.csv')
            if os.path.exists(log_run_sequence_time_csv):
                log_run_sequence_time_df = pd.read_csv(log_run_sequence_time_csv)

            for system_output in system_outputs:
                system_output_idx = get_idx(system_output)
                if system_output_idx > idx_last_trajectory:
                    system_output_idx_string = get_idx_string(system_output)
                    print(f"    Removing: {os.path.join(sequence_path, f'*{system_output_idx_string}*')}")
                    idx_files = [item for item in items if os.path.isfile(os.path.join(sequence_path, item)) and system_output_idx_string in item]
                    for idx_file in idx_files:
                        os.remove(os.path.join(sequence_path, idx_file))

                    if os.path.exists(log_ablation_parameters_csv):
                        log_ablation_parameters_df = log_ablation_parameters_df[log_ablation_parameters_df['expId'] != system_output_idx]
                    if os.path.exists(log_run_sequence_time_csv):
                        log_run_sequence_time_df = log_run_sequence_time_df[log_run_sequence_time_df['experiment_id'] != system_output_idx]

            if os.path.exists(log_ablation_parameters_csv):
                log_ablation_parameters_df.to_csv(log_ablation_parameters_csv, index=False)
            if os.path.exists(log_run_sequence_time_csv):
                log_run_sequence_time_df.to_csv(log_run_sequence_time_csv, index=False)

            evaluation_folders = [item for item in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, item)) and "evaluation" in item]
            for evaluation_folder in evaluation_folders:
               shutil.rmtree(os.path.join(sequence_path, evaluation_folder))
        else:
            print(f"\n{SCRIPT_LABEL}{GREEN}Succesful sequence:{RESET} {sequence_path}")