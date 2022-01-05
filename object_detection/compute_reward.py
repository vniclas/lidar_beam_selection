import glob
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np

from rl_l2o.compute_reward import RewardComputer as RewardComputerBase

# This number should match the value of NUM_EPOCHS in the corresponding detector, e.g,
# third_party/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml
DETECTOR_NUM_EPOCHS = 40


class RewardComputerObjectDetection(RewardComputerBase):
    DEFAULT_CONFIG = {
        'plv2_dir': '',
        'opcd_dir': '',
        'kitti_dir': '',
        'output_dir': '',
        'pred_path': '',
        'detector': 'pointpillar',
        'num_threads': 24
    }

    @staticmethod
    def compute_reward(
            beams: list,  # List of beam ids
            plv2_dir: str,  # PseudoLidarv2/gdc directory
            opcd_dir: str,  # OpenPCDet directory
            kitti_dir: str,  # KITTI dataset directory
            output_dir: str,  # Output directory
            pred_path: str,  # Path of SDN predictions
            detector: str = 'pointpillar',  # pointpillar or pointrcnn
            num_threads: int = 32) -> np.ndarray:
        assert len(beams) == 4, 'This implementation assumes exactly 4 beams.'
        assert detector in ['pointpillar', 'pointrcnn']

        # Get paths
        calib_path = os.path.join(kitti_dir, 'calib')
        image_path = os.path.join(kitti_dir, 'image_2')
        point_path = os.path.join(kitti_dir, 'velodyne_original')
        split_file = os.path.join(plv2_dir, 'image_sets/trainval.txt')
        tmp_dir = os.path.join(output_dir, f'tmp_dir_{detector}')

        tag = f'{beams[0]}_{beams[1]}_{beams[2]}_{beams[3]}'
        train_dir = os.path.join(opcd_dir, 'output', 'kitti_models', detector, tag)

        # If the results file already exists, skip the computation
        reward = _check_for_existing_logs(train_dir)
        if reward is not None:
            return reward

        # If the pseudo lidar data already exists, skip most of the pre-processing
        output_name = f'pseudo_lidar_{beams[0]}_{beams[1]}_{beams[2]}_{beams[3]}'
        output_path = os.path.join(output_dir, output_name)
        if not _does_data_exist(output_path):

            # Simulate 4 beam lidar
            print(f'Simulating 4 beam lidar with beam selection: {beams}...')
            output_name = f'velodyne_{beams[0]}_{beams[1]}_{beams[2]}_{beams[3]}'
            output_path = os.path.join(output_dir, output_name)
            velodyne_4_beam = f' \
                python {plv2_dir}/sparsify.py \
                --calib_path {calib_path} \
                --image_path {image_path} \
                --split_file {split_file} \
                --ptc_path {point_path} \
                --W 1024 \
                --H 64 \
                --line_spec {beams[0]} {beams[1]} {beams[2]} {beams[3]} \
                --output_path {output_path} \
                --store_line_map_dir {tmp_dir} \
                --threads {num_threads} \
            '

            if not _does_data_exist(output_path):
                os.system(velodyne_4_beam)

            # Get ground truth depth map from original 4 beams
            print('Generating ground truth depth maps from the simulated 4 beam lidar...')
            input_path = output_path
            output_name = f'gt_depthmap_{beams[0]}_{beams[1]}_{beams[2]}_{beams[3]}'
            output_path = os.path.join(output_dir, output_name)
            gt_depthmap_4_beam = f' \
                python {plv2_dir}/ptc2depthmap.py \
                --output_path {output_path} \
                --input_path {input_path} \
                --calib_path {calib_path} \
                --image_path {image_path} \
                --split_file {split_file}  \
                --threads {num_threads} \
            '

            if not _does_data_exist(output_path):
                os.system(gt_depthmap_4_beam)

            # Run batch gdc using ground truth 4 beams on predicted depth maps
            print('Running batch GDC using 4 beam ground truth depth map on predicted depth maps...')
            input_path = output_path
            output_name = f'gdc_depthmap_{beams[0]}_{beams[1]}_{beams[2]}_{beams[3]}'
            output_path = os.path.join(output_dir, output_name)
            gdc_depthmap_4_beam = f' \
                python {plv2_dir}/main_batch.py \
                --output_path {output_path} \
                --input_path {pred_path} \
                --calib_path {calib_path} \
                --gt_depthmap_path {input_path} \
                --threads {num_threads} \
                --split_file {split_file} \
            '

            if not _does_data_exist(output_path):
                os.system(gdc_depthmap_4_beam)

            # Get pseudo lidar from corrected depth
            print('Generating pseudo lidar point clouds from corrected depth maps...')
            input_path = output_path
            output_name = f'pseudo_lidar_{beams[0]}_{beams[1]}_{beams[2]}_{beams[3]}'
            output_path = os.path.join(output_dir, output_name)
            pseudo_lidar_4_beam = f' \
                python {plv2_dir}/depthmap2ptc.py \
                --output_path {output_path} \
                --input_path {input_path} \
                --calib_path {calib_path} \
                --threads {num_threads} \
                --split_file {split_file} \
            '

            if not _does_data_exist(output_path):
                os.system(pseudo_lidar_4_beam)

        # Sparsify to 64 lines
        print('Sparsifying pseudo lidar point cloud to 64 beams...')
        input_path = output_path
        output_path = os.path.join(kitti_dir, 'velodyne')
        sparse_pseudo_lidar_4_beam = f' \
            python {plv2_dir}/sparsify.py \
            --output_path {output_path} \
            --calib_path {calib_path} \
            --image_path {image_path} \
            --ptc_path {input_path} \
            --split_file {split_file} \
            --W 1024 --slice 1 --H 64 \
            --threads {num_threads} \
        '

        os.system(sparse_pseudo_lidar_4_beam)

        # This only needs to be done once
        # Generate info files and ground truth database for KITTI
        if not os.path.exists(f'{opcd_dir}/data/kitti/kitti_infos_trainval.pkl'):
            print('Generating KITTI file database...')
            kitti_database = f' \
                python -m pcdet.datasets.kitti.kitti_dataset \
                create_kitti_infos \
                {opcd_dir}/tools/cfgs/dataset_configs/kitti_dataset.yaml \
            '

            os.system(kitti_database)

        # Start training the detector
        print('Submitting training job...')
        subprocess.Popen([str(Path(__file__).parent / 'scripts' / f'train_{detector}.sh'), tag])

        # Check if job has started and look for log file
        reward = _check_for_existing_logs(train_dir, frequency=5)
        return reward


def _check_for_existing_logs(train_dir: str, frequency: int = -1) -> Optional[np.ndarray]:
    if frequency > 0:
        print(f'Looking for job every {frequency} seconds in {train_dir}', end='', flush=True)
    log_file_path = None
    while True:
        for file_path in glob.glob(os.path.join(train_dir, 'log_train*.txt')):
            if os.path.getsize(file_path) > 0:
                log_file_path = file_path
                break
        if log_file_path is None and frequency > 0:
            time.sleep(frequency)
        elif log_file_path is not None:
            print(f'Log file located: {log_file_path}.\nChecking for performance of epoch {DETECTOR_NUM_EPOCHS}...')
            break
        else:
            return None

    # Continuously check log file for evaluation
    check_phrase = f'Performance of EPOCH {DETECTOR_NUM_EPOCHS}'
    check_index = -1
    car_ap = None
    pedestrian_ap = None
    cyclist_ap = None
    while True:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if check_phrase in line:
                    check_index = i
                elif check_index == -1:
                    continue
                if i == check_index + 12:  # 3D Car AP
                    car_ap = line
                elif i == check_index + 32:  # 3D Pedestrian AP
                    pedestrian_ap = line
                elif i == check_index + 52:  # 3D Cyclist AP
                    cyclist_ap = line
                    break
        if car_ap is not None and pedestrian_ap is not None and cyclist_ap is not None:
            break
        if frequency > 0:
            print('.', end='', flush=True)
            time.sleep(30)

    assert '3d   AP:' in car_ap, car_ap
    assert '3d   AP:' in pedestrian_ap, pedestrian_ap
    assert '3d   AP:' in cyclist_ap, cyclist_ap
    result = np.array([float(car_ap[8:-1].split(', ')[1]) / 100])  # 3D AP for car: moderate
    # result = np.array([float(j) / 100 for j in car_ap[8:-1].split(', ')])  # Get 3D AP for easy, medium, and hard
    # result = np.array([
    #     float(car_ap[8:-1].split(', ')[1]) / 100,
    #     float(pedestrian_ap[8:-1].split(', ')[1]) / 100,
    #     float(cyclist_ap[8:-1].split(', ')[1]) / 100
    # ])  # Get 3D AP for car, ped, and cyclists of moderate difficulty
    return result


def _does_data_exist(path: str, number_expected_files: int = 7481) -> bool:
    if not os.path.exists(path):
        return False
    number_files = len(os.listdir(path))
    if number_files == number_expected_files:
        return True
    return False
