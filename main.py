from pathlib import Path

from object_detection.compute_reward import RewardComputerObjectDetection
from rl_l2o.eps_greedy_search import EpsGreedySearch

detector = 'pointpillar'
# detector = 'rcnn'

# Overwrite default config parameters
config = {
    'checkpoint_files_path': Path(__file__).absolute().parent / f'checkpoints_{detector}',
    'compute_reward': {
        'plv2_dir': '/home/USER/git/lidar_beam_selection/third_party/Pseudo_Lidar_V2/gdc',
        'opcd_dir': '/home/USER/git/lidar_beam_selection/third_party/OpenPCDet',
        'kitti_dir': '/home/USER/data/kitti/training',
        'output_dir': '/home/USER/data/pseudo_lidar_v2',
        'pred_path': '/home/USER/data/pseudo_lidar_v2/sdn_kitti_train_set/depth_maps/trainval',
        'detector': detector,
    }
}
logfile = Path(__file__).parent / f'log_{detector}.txt'

features_pcl_file = Path(__file__).absolute().parent / 'object_detection' / 'data' / 'features_pcl.pkl'
reward_computer = RewardComputerObjectDetection(config['compute_reward'])

# To start a new run from scratch
search = EpsGreedySearch(features_pcl_file, reward_computer, config, logfile)

# To resume a previous run
# checkpoint_file = Path(__file__).parent / f'checkpoints_{detector}' / 'checkpoint_100.pkl'
# search.load_checkpoint(checkpoint_file, continue_searching=True)

# To load the cache containing the true reward
# search.load_reward_computation_cache(checkpoint_file)  # To load the cache containing the true reward

# Start the search
search.run()

# Display the best beam configuration
best_state = search.best_state(return_reward=True)
print(f'\033[96m RESULT: state={best_state[0]}, reward={best_state[1]:.3f} \033[0m')  # Colored as OKCYAN
