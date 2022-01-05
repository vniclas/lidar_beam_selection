import pickle
import random
from os import PathLike
from typing import Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from sympy.utilities.iterables import multiset_permutations
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], features_pcl: np.ndarray, elevation_angles_deg: np.ndarray):
        super().__init__()
        self.features_pcl = features_pcl.astype(np.float32)
        self.elevation_angles_deg = elevation_angles_deg.astype(np.float32)
        self.state = data['state'].astype(np.int)
        self.reward = data['reward'].astype(np.float32)

    def __len__(self):
        return self.state.shape[0]

    def __getitem__(self, item):
        state = self.state[item, :]
        reward = self.reward[item, :]

        features_pcl = self.features_pcl[state]
        pairwise_features = np.c_[np.diff(self.elevation_angles_deg[np.sort(state)])]  # permutation invariant
        features_pcl = np.r_[features_pcl.flatten(), pairwise_features.flatten()]

        sample = {'state': state, 'features_pcl': features_pcl, 'reward': reward}
        return sample


class MyNetwork(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.INPUT_SHAPE = 39
        self.output_shape = output_shape

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.INPUT_SHAPE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_shape),
            nn.Sigmoid()  # Squeeze between 0 and 1 since it is accuracy
        )

    def forward(self, x):
        return self.net(x)


class RewardPredictor:
    DEFAULT_CONFIG = {'num_training_epochs': 10, 'training_batch_size': 8, 'device': None}

    def __init__(self,
                 num_selected_beams: int,
                 num_reward_signals: int,
                 features_pcl_file: Union[str, PathLike],
                 config: Optional[Dict[str, Any]] = None):
        self.config = RewardPredictor.DEFAULT_CONFIG
        if config:
            self.config.update(config)
        self.num_selected_beams = num_selected_beams
        self.num_reward_signals = num_reward_signals
        self.features_pcl = None
        self.elevation_angles_deg = None
        self._load_features(features_pcl_file)

        if self.config['device'] is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.config['device'] == 'cpu':
            self.device = torch.device('cpu')
        elif self.config['device'] == 'cuda':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                raise RuntimeError('CUDA runtime not available.')
        else:
            raise ValueError(f'Invalid device: {self.config["device"]}. Valid options are: ["cpu", "cuda"].')
        print(f'Found device: {self.device}')

        self.regressor = MyNetwork(self.num_reward_signals)
        self.regressor.to(self.device)
        self.training_data = {
            'state': np.empty((0, self.num_selected_beams), dtype=np.int),
            'reward': np.empty((0, self.num_reward_signals), dtype=np.float32)
        }

    def load_training_data(self, training_data: Dict[str, np.ndarray]):
        self.training_data = training_data

    def add_to_training_data(self, state: np.ndarray, reward: np.ndarray):
        assert state.shape == (self.num_selected_beams,)
        assert reward.shape == (self.num_reward_signals,)

        if not self._is_in_training_data(state):
            self.training_data['state'] = np.append(self.training_data['state'],
                                                    np.reshape(np.sort(state), (1, self.num_selected_beams)),
                                                    axis=0)
            self.training_data['reward'] = np.append(self.training_data['reward'],
                                                     np.reshape(reward, (1, self.num_reward_signals)),
                                                     axis=0)

    def train(self, permute: bool = False, reset_weights: bool = True, verbose: bool = False):
        if reset_weights:
            self.regressor = MyNetwork(self.num_reward_signals)
            self.regressor.to(self.device)

        states, rewards = self._permute_training_data(permute)
        training_dataset = MyDataset({'state': states, 'reward': rewards}, self.features_pcl, self.elevation_angles_deg)
        training_dataloader = DataLoader(training_dataset, batch_size=self.config['training_batch_size'], shuffle=True)

        criterion = nn.MSELoss(reduction='sum')
        optimizer = optim.Adam(self.regressor.parameters(), lr=.001, weight_decay=.0001)

        self.regressor.train()
        with tqdm(training_dataloader,
                  disable=not verbose,
                  unit='batch',
                  total=self.config['num_training_epochs'] * len(training_dataloader),
                  desc=f'Training | {self.config["num_training_epochs"]} epochs') as pbar:
            epoch_loss = 0
            for epoch in range(self.config['num_training_epochs']):
                train_loss = 0
                for sample_batched in training_dataloader:
                    optimizer.zero_grad()
                    features_pcl = sample_batched['features_pcl'].to(self.device)
                    true_reward = sample_batched['reward'].to(self.device)
                    predicted_reward = self.regressor(features_pcl)
                    loss = criterion(predicted_reward, true_reward)

                    train_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix(epoch=epoch, epoch_loss=epoch_loss, batch_loss=loss.item())
                    pbar.update(1)
                epoch_loss = train_loss / len(training_dataloader)

    def predict(self, state: np.ndarray) -> float:
        self.regressor.eval()
        with torch.no_grad():
            features_pcl = self._state_to_features(state)
            features_pcl = torch.from_numpy(features_pcl).to(self.device)
            predicted_reward = self.regressor(features_pcl).cpu().detach().numpy()
        return float(predicted_reward.mean())

    def _is_in_training_data(self, state: np.ndarray):
        assert state.shape == (self.num_selected_beams,)
        return (np.sort(state) == np.sort(self.training_data['state'])).all(axis=1).any()

    def _permute_training_data(self, permute: bool = True, in_place: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if permute:
            state, reward = permute_dataset(self.training_data['state'], self.training_data['reward'])
            if in_place:
                self.training_data['state'], self.training_data['reward'] = state, reward
        else:
            state, reward = self.training_data['state'], self.training_data['reward']
        return state, reward

    def _load_features(self, features_pcl_file: Union[str, PathLike]):
        with open(features_pcl_file, 'rb') as f:
            features_pcl_raw = pickle.load(f)

        features_pcl = np.c_[np.fromiter(features_pcl_raw['number_points_avg'].values(), dtype=np.float32),
                             np.fromiter(features_pcl_raw['mean_d_avg'].values(), dtype=np.float32),
                             np.fromiter(features_pcl_raw['mean_d_std_avg'].values(), dtype=np.float32),
                             np.array([
                                 features_pcl_raw['semantic_classes_avg'][beam_id]
                                 for beam_id in np.arange(0, len(features_pcl_raw['semantic_classes_avg']))
                             ]) / np.fromiter(features_pcl_raw['number_points_avg'].values(), dtype=np.float32).reshape(
                                 (-1, 1))].astype(np.float32)
        self.elevation_angles_deg = np.fromiter(features_pcl_raw['elevation_angles_deg'].values(), dtype=np.float32)

        # Standardize features
        features_pcl = (features_pcl - features_pcl.mean(axis=0)) / features_pcl.std(axis=0)
        features_pcl = np.c_[features_pcl, self.elevation_angles_deg]

        # Save for future usage
        self.features_pcl = features_pcl

    def _state_to_features(self, state: np.ndarray):
        assert state.shape == (self.num_selected_beams,)
        features_pcl = self.features_pcl[state]
        pairwise_features = np.c_[np.diff(self.elevation_angles_deg[np.sort(state)])]  # permutation invariant
        features = np.r_[features_pcl.flatten(), pairwise_features.flatten()]
        return features.reshape((1, -1))


def permute_dataset(state, metrics):
    active_beams_ret = []
    metrics_ret = []

    for b, m in zip(state, metrics):
        for p in multiset_permutations(b):
            active_beams_ret.append(p)
            metrics_ret.append(m)

    tmp = list(zip(active_beams_ret, metrics_ret))
    random.shuffle(tmp)
    active_beams_ret, metrics_ret = zip(*tmp)

    active_beams_ret = np.vstack(active_beams_ret)
    metrics_ret = np.vstack(metrics_ret)
    return active_beams_ret, metrics_ret
