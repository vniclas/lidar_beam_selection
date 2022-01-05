#!/bin/bash

cd /home/USER/git/lidar_beam_selection/third_party/OpenPCDet/tools || exit

bash scripts/dist_train.sh 4 --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt_save_interval 40 --extra_tag $1
