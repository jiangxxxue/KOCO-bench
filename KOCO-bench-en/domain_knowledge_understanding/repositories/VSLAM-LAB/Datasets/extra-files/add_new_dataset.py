import os
import sys
import shutil
import argparse

sys.path.append(os.getcwd())

from path_constants import VSLAM_LAB_DIR
from path_constants import VSLAMLAB_BENCHMARK
from utilities import check_sequence_integrity
from utilities import ws

# This script integrates an EXISTING and COMPLIANT dataset into VSLAM-LAB by creating a new dataset class: dataset_{
# dataset_name}.py, generating a corresponding YAML configuration: dataset_{dataset_name}.yaml, and updating imports
# in: dataset_utilities.py.
def main():
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('dataset_name', type=str, help=f"Name of the dataset")
    args = parser.parse_args()

    dataset_name = str(args.dataset_name).lower()
    DATASET_NAME = str(args.dataset_name).upper()

    # Verify the existence of the dataset folder
    dataset_path = os.path.join(VSLAMLAB_BENCHMARK, DATASET_NAME)
    if not os.path.exists(dataset_path):
        print(f"Dataset \'{dataset_path}\' does not exist!")
        exit(0)
    print(f"Dataset '{dataset_path}' is available.")

    # Validate the integrity of files in the sequences
    sequence_paths = os.listdir(dataset_path)
    sequence_names = [sequence_path for sequence_path in sequence_paths if
                      os.path.isdir(os.path.join(dataset_path, sequence_path))]

    for sequence_name in sequence_names:
        sequence_path = os.path.join(dataset_path, sequence_name)
        if os.path.exists(sequence_path):
            sequence_complete = check_sequence_integrity(dataset_path, sequence_name, True)
            if sequence_complete:
                print(f"{ws(4)}Sequence '{sequence_name}' is available.")
            else:
                print(f"{ws(4)}Some files in sequence {sequence_name} are corrupted.")
                print(f"{ws(4)}Removing and downloading again sequence {sequence_name} ")
                print(f"{ws(4)}THIS PART OF THE CODE IS NOT YET IMPLEMENTED. REMOVE THE FILES MANUALLY")
                sys.exit(1)

    # Derive a new class: dataset_{dataset_name}.py
    dataset_py_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', f"dataset_{dataset_name}.py")
    print(f"Derived a new class: {dataset_py_file}")
    shutil.copy(os.path.join(VSLAM_LAB_DIR, 'Datasets', 'extraFiles', 'dataset_template.py'), dataset_py_file)
    replace_variable_in_script(dataset_py_file, 'dataset_name_template', dataset_name)
    replace_variable_in_script(dataset_py_file, 'DATASET_NAME_TEMPLATE', f"{DATASET_NAME}")

    # Create a YAML configuration: dataset_{dataset_name}.yaml
    dataset_yaml_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', f"dataset_{dataset_name}.yaml")
    print(f"Created a YAML configuration: {dataset_yaml_file}")
    yaml_content_lines = [
        f"dataset_name : {dataset_name} \n",
        f"rgb_hz: {30.0}",
        f"url_download_root: \"\"\n",
        f"sequence_names:"
    ]

    for sequence_name in sequence_names:
        yaml_content_lines.append("  - " + sequence_name)

    with open(dataset_yaml_file, 'w') as file:
        for line in yaml_content_lines:
            file.write(f"{line}\n")

    # Add imports to: dataset_utilities.py
    dataset_utilities_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', "dataset_utilities.py")
    print(f"Added imports to : {dataset_utilities_file}")
    comment = "# ADD your imports here"
    statement = f"from Datasets.dataset_{dataset_name} import {DATASET_NAME}_dataset\n"
    add_import_statement(dataset_utilities_file, statement, comment)

    comment = "# ADD your datasets here"
    statement = f"        \"{dataset_name}\": lambda: {DATASET_NAME}_dataset(benchmark_path),\n"
    dataset_utilities_file = os.path.join(VSLAM_LAB_DIR, 'Datasets', "dataset_utilities.py")
    add_import_statement(dataset_utilities_file, statement, comment)


# Replace all occurrences of old_variable with new_variable in the specified file
def replace_variable_in_script(file_path, old_variable, new_variable):
    with open(file_path, 'r') as file:
        file_content = file.read()

    updated_content = file_content.replace(old_variable, new_variable)

    with open(file_path, 'w') as file:
        file.write(updated_content)


# Add the import statement below the specified comment if it doesn't already exist in the file
def add_import_statement(file_path, statement, comment):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if statement in lines:
        return

    for i, line in enumerate(lines):
        if comment in line:
            lines.insert(i + 1, statement)
            break

    with open(file_path, 'w') as file:
        file.writelines(lines)


if __name__ == "__main__":
    main()
