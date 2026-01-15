"""
Module: VSLAM-LAB - Evaluate - metrics.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2024-07-12
- Updated: 2024-07-12
- License: GPLv3 License
- List of Known Dependencies;
    * The aligning functions in this script are a modified version of the code in Useful tools for the RGB-D benchmark:
    https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools#evaluation

"""

import numpy as np


def align_trajectory_with_groundtruth(trajectory_txt, groundtruth_txt, max_time_difference, perc=1.0, iPerc=1,
                                      time_offset=0.0):
    """Align one trajectory to a ground truth trajectory using the method of Horn (closed-form).
    The matches between trajectories are computed based on the timestamps

    Input:
    trajectory_txt  -- a text file containing the trajectory data  : timestamp x y z qx qy qz qw (Nx8)
    groundtruth_txt -- a text file containing the ground truth data: timestamp x y z qx qy qz qw (Nx8)
    max_time_difference -- the maximum time difference between the trajectory and ground truth trajectory timestamps
    time_offset -- the time offset between the trajectory and ground truth timestamps

    Output:
    traj_xyz_aligned      -- sampled trajectory aligned : x y z , (nx3) numpy array type
    gt_xyz                -- sampled ground truth       : x y z , (nx3) numpy array type
    traj_xyz_full_aligned -- full trajectory aligned    : x y z , (Nx3) numpy array type
    gt_xyz_full           -- full ground truth          : x y z , (Nx3) numpy array type

    """

    # Load trajectories from text files.
    gt_list = read_file_list(groundtruth_txt)
    traj_list = read_file_list(trajectory_txt)

    gt_stamps = gt_list.keys()
    gt_stamps = list(gt_stamps)
    gt_stamps.sort()

    traj_stamps = traj_list.keys()
    traj_stamps = list(traj_stamps)
    traj_stamps.sort()

    gt_xyz_full = np.matrix([[float(value) for value in gt_list[b][0:3]] for b in gt_stamps]).transpose()
    traj_xyz_full = np.matrix([[float(value) for value in traj_list[b][0:3]] for b in traj_stamps]).transpose()

    # Associate trajectories by looking for matches between the timestamps.
    matches = associate(gt_list, traj_list, float(time_offset), float(max_time_difference))

    gt_xyz = np.matrix([[float(value) for value in gt_list[a][0:3]] for a, b in matches]).transpose()
    traj_xyz = np.matrix([[float(value) for value in traj_list[b][0:3]] for a, b in matches]).transpose()

    # Align trajectories using the Horn method
    rot, trans, trans_error, scale = align_horn(traj_xyz, gt_xyz)
    traj_xyz_aligned = scale * rot * traj_xyz + trans
    traj_xyz_full_aligned = scale * rot * traj_xyz_full + trans

    traj_xyz_aligned = np.asarray(traj_xyz_aligned.transpose())
    gt_xyz = np.asarray(gt_xyz.transpose())

    traj_xyz_full_aligned = np.asarray(traj_xyz_full_aligned.transpose())
    gt_xyz_full = np.asarray(gt_xyz_full.transpose())
    return traj_xyz_aligned, gt_xyz, traj_xyz_full_aligned, gt_xyz_full

def align_horn(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)

    """
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered

    dots = 0.0
    norms = 0.0
    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(np.sum(dots) / np.sum(norms))

    trans = data.mean(1) - s * rot * model.mean(1)

    model_aligned = s * rot * model + trans
    alignment_error = model_aligned - data

    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, trans, trans_error, s

def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples

    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
            len(line) > 0 and line[0] != "#"]
    list = [(float(l[0]), l[1:]) for l in list if len(l) > 1]
    return dict(list)


def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
    to find the closest match for every input tuple.

    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b)
                         for a in first_keys
                         for b in second_keys
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys = list(first_keys)
            first_keys.remove(a)
            second_keys = list(second_keys)
            second_keys.remove(b)
            matches.append((a, b))

    matches.sort()
    return matches
