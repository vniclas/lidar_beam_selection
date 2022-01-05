import pickle
from copy import deepcopy
from datetime import datetime
from itertools import product
from os import PathLike
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

from rl_l2o.compute_reward import RewardComputer
from rl_l2o.predict_reward import RewardPredictor
from rl_l2o.utils import Colors


class EpsGreedySearch:
    DEFAULT_CONFIG = {
        'initial_beam_ids': None,  # after initialization, start with these beams. If None, start with the best so far.
        'epsilon': .1,  # take a random action with this probability
        'num_initial_samples': 5,  # number of random samples to start training the value function predictor
        'num_samples': 200,  # total number of samples
        'max_step_size': 2,  # the beam IDs can be shifted a maximum of this number (pos/neg)
        'seed': None,  # used to initialize all random number generators
        'num_selected_beams': 4,  # number of beams selected for valid states
        'num_reward_signals': 1,  # number of environmental signals used to predict the overall reward
        'min_beam_id': 1,  # smallest valid beam ID
        'max_beam_id': 40,  # highest valid beam ID
        'checkpoint_files_path': Path(__file__).absolute().parents[1] / 'checkpoints',
        'predict_reward': {
            'num_training_epochs': 10,
            'training_batch_size': 8,
            'device': None
        },
    }

    def __init__(
            self,
            features_pcl_file: Union[str, PathLike],
            reward_computer: RewardComputer,
            config: Optional[Dict[str, Any]] = None,
            logfile: Union[str, PathLike] = 'log.txt',
    ):
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        if self.config['initial_beam_ids'] is not None:
            assert len(self.config['initial_beam_ids']) == self.config['num_selected_beams']
            assert min(self.config['initial_beam_ids']) >= self.config['min_beam_id']
            assert max(self.config['initial_beam_ids']) <= self.config['max_beam_id']
        self.config['features_pcl_file'] = str(features_pcl_file)

        # Random number generators
        self.epsilon_generator = np.random.default_rng(seed=self.config['seed'])
        self.random_action_generator = np.random.default_rng(seed=self.config['seed'])
        self.random_state_generator = np.random.default_rng(seed=self.config['seed'])

        # Miscellaneous objects
        self.reward_predictor = RewardPredictor(self.config['num_selected_beams'], self.config['num_reward_signals'],
                                                self.config['features_pcl_file'], self.config['predict_reward'])
        self.reward_computer = reward_computer
        self.reward_computer.set_number_reward_signals(self.config['num_reward_signals'])
        self.log_history = {}  # step_counter: (message, color)
        self.state_reward_history = []  # (beams, reward, predicted_reward)
        self.state_action_pairs = []  # (beams, steps)
        self.step_counter = 0
        self._continue_searching = False  # can be set when loading a checkpoint

        # Logging
        self.logfile = Path(logfile)
        if self.logfile.exists():
            prev_logfile = str(logfile).replace(self.logfile.suffix, f'_old{self.logfile.suffix}')
            self.logfile.rename(prev_logfile)
            print(f'{Colors.WARNING}WARNING: Found existing logfile and renamed it to "{prev_logfile}".{Colors.ENDC}')

    def run(self) -> np.ndarray:
        self._initialize_predictor()
        if self._continue_searching:
            state = self.state_reward_history[-1][0]
        elif self.config['initial_beam_ids'] is None:
            state = self.best_state()
        else:
            if self.step_counter < self.config['num_samples']:
                self.step_counter += 1
                state = np.asarray(self.config['initial_beam_ids'], dtype=np.int)
                reward, reward_signals = self.reward_computer.compute(state)
                self.reward_predictor.add_to_training_data(state, reward_signals)
                self.reward_predictor.train(permute=True)
                is_best_state = self._add_to_state_reward_history(state, reward)
                self._write_to_log(state, reward, is_best_state=is_best_state)
                self.save_checkpoint()

        # Main loop
        while self.step_counter < self.config['num_samples']:
            self.step_counter += 1

            if self.epsilon_generator.random() < self.config['epsilon']:
                action = self._get_random_action(state)
                predicted_reward = self.reward_predictor.predict(state)
            else:
                action, predicted_reward = self._get_best_action(state)
            state = self._apply_action(state, action)
            assert self._is_valid_state(state), state
            if self._is_in_state_action_pairs(state, action):
                state = self._get_random_state(self.random_state_generator)
                self._write_message_to_log(
                    'WARNING: Resampled new random state due to a loop in the state-action-history.', Colors.WARNING,
                    True)

            reward, reward_signals = self.reward_computer.compute(state)
            self.reward_predictor.add_to_training_data(state, reward_signals)
            self.reward_predictor.train(permute=True)

            is_best_state = self._add_to_state_reward_history(state, reward, predicted_reward)
            self._write_to_log(state, reward, predicted_reward, is_best_state)
            self.save_checkpoint()

        return self.best_state()

    def best_state(self, return_reward: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        state_reward = np.array([[x[0], x[1]] for x in self.state_reward_history], dtype=object)
        best_reward_index = int(np.argmax(state_reward[:, 1]))
        best_state_reward = self.state_reward_history[best_reward_index]
        best_state_reward = (np.sort(best_state_reward[0]), best_state_reward[1])
        if return_reward:
            return best_state_reward
        return best_state_reward[0]

    def save_checkpoint(self):
        if not self.config['checkpoint_files_path'].exists():
            self.config['checkpoint_files_path'].mkdir(parents=True, exist_ok=True)
        checkpoint_file = self.config['checkpoint_files_path'] / f'checkpoint_{str(self.step_counter).zfill(3)}.pkl'
        checkpoint = {
            'config': self.config,
            'state_reward_history': self.state_reward_history,
            'state_action_pairs': self.state_action_pairs,
            'step_counter': self.step_counter,
            'log_history': self.log_history,
            'epsilon_generator': self.epsilon_generator.__getstate__(),
            'random_action_generator': self.random_action_generator.__getstate__(),
            'random_state_generator': self.random_state_generator.__getstate__(),
            'reward_computer_cache': self.reward_computer.cache,
            'reward_predictor_training_data': self.reward_predictor.training_data,
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f, pickle.DEFAULT_PROTOCOL)

    def load_checkpoint(self, checkpoint_file: Union[str, PathLike], continue_searching: bool = True):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
        self._continue_searching = continue_searching

        # Check for compatible setups
        assert self.config['num_selected_beams'] == checkpoint['config']['num_selected_beams']

        # We do not load paths and total number of samples from checkpoint files
        config = deepcopy(self.config)
        self.config = checkpoint['config']
        self.config['compute_reward'] = config['compute_reward']
        self.config['checkpoint_files_path'] = config['checkpoint_files_path']
        self.config['features_pcl_file'] = config['features_pcl_file']
        self.config['num_samples'] = config['num_samples']

        self.state_action_pairs = checkpoint['state_action_pairs']
        self.step_counter = checkpoint['step_counter']
        if 'log_history' in checkpoint:
            self.log_history = checkpoint['log_history']
        self.epsilon_generator.__setstate__(checkpoint['epsilon_generator'])
        self.random_action_generator.__setstate__(checkpoint['random_action_generator'])
        self.random_state_generator.__setstate__(checkpoint['random_state_generator'])

        self.reward_predictor = RewardPredictor(self.config['num_selected_beams'], self.config['num_reward_signals'],
                                                self.config['features_pcl_file'], self.config['predict_reward'])
        self.reward_predictor.load_training_data(checkpoint['reward_predictor_training_data'])
        self.reward_computer.load_cache(checkpoint['reward_computer_cache'])

        for step_counter_replay, state_reward in enumerate(checkpoint['state_reward_history'], start=1):
            if step_counter_replay in self.log_history:
                self._write_message_to_log(self.log_history[step_counter_replay][0],
                                           self.log_history[step_counter_replay][1])
            if len(state_reward) == 2:
                state_reward = (state_reward[0], state_reward[1], None)
            is_best_state = self._add_to_state_reward_history(state_reward[0], state_reward[1], state_reward[2])
            self._write_to_log(state_reward[0], state_reward[1], state_reward[2], is_best_state, step_counter_replay)

        self._write_message_to_log('INFO: Resuming from a checkpoint.', Colors.OKBLUE, True)

    def load_reward_prediction_cache(self, checkpoint_file: Union[str, PathLike]):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)

        # Check for compatible setups
        assert self.config['num_selected_beams'] == checkpoint['config']['num_selected_beams']

        self.reward_predictor.load_training_data(checkpoint['reward_predictor_training_data'])

    def load_reward_computation_cache(self, checkpoint_file: Union[str, PathLike]):
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)

        # Check for compatible setups
        assert self.config['num_selected_beams'] == checkpoint['config']['num_selected_beams']

        self.reward_computer.load_cache(checkpoint['reward_computer_cache'])

    def _initialize_predictor(self):
        while self.step_counter < self.config['num_initial_samples']:
            self.step_counter += 1
            state = self._get_random_state(self.random_state_generator)
            reward, reward_signals = self.reward_computer.compute(state)
            self.reward_predictor.add_to_training_data(state, reward_signals)
            is_best_state = self._add_to_state_reward_history(state, reward)
            self._write_to_log(state, reward, is_best_state=is_best_state)
            self.save_checkpoint()
        self.reward_predictor.train(permute=True)

    def _is_valid_state(self, state: np.ndarray) -> bool:
        check = np.unique(state).size == self.config['num_selected_beams']
        check &= state.min() >= self.config['min_beam_id']
        check &= state.max() <= self.config['max_beam_id']
        return check

    def _get_random_state(self, random_number_generator: np.random.Generator) -> np.ndarray:
        state = np.empty((1,))
        while not self._is_valid_state(state):
            state = np.unique(
                np.sort(
                    random_number_generator.integers(self.config['min_beam_id'],
                                                     self.config['max_beam_id'],
                                                     endpoint=True,
                                                     size=(self.config['num_selected_beams'],))))
        return state

    def _get_random_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        action = self.random_action_generator.integers(-self.config['max_step_size'],
                                                       self.config['max_step_size'],
                                                       endpoint=True,
                                                       size=(self.config['num_selected_beams'],))
        # If a state is provided, only valid actions will be returned.
        while state is not None and not self._is_valid_state(self._apply_action(state, action)):
            action = self.random_action_generator.integers(-self.config['max_step_size'],
                                                           self.config['max_step_size'],
                                                           endpoint=True,
                                                           size=(self.config['num_selected_beams'],))
        return action

    def _get_best_action(self, state: np.ndarray, show_pbar: bool = False) -> Tuple[np.ndarray, Optional[float]]:
        number_selected_beams = self.config['num_selected_beams']
        actions = []
        rewards = []
        valid_steps = list(range(-self.config['max_step_size'], self.config['max_step_size'] + 1))
        with tqdm(total=len(valid_steps) ** number_selected_beams, desc='Predicting best action',
                  disable=not show_pbar) as pbar:
            for action in product(*[valid_steps] * number_selected_beams):
                action = np.array(action)
                new_state = self._apply_action(state, action)
                if self._is_valid_state(new_state):
                    predicted_reward = self.reward_predictor.predict(new_state)
                    actions.append(action)
                    rewards.append(predicted_reward)
                pbar.update(1)
        if rewards:
            best_reward = max(rewards)
            best_action = actions[rewards.index(best_reward)]
        else:
            best_action = np.zeros((number_selected_beams,), dtype=np.int)
            best_reward = None
        return best_action, best_reward

    @staticmethod
    def _apply_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
        new_state = state + action
        return new_state

    def _is_in_state_action_pairs(self, state: np.ndarray, action: np.ndarray) -> bool:
        sorted_action = action[np.argsort(state)]
        sorted_state = np.sort(state)
        for state_action_pair in self.state_action_pairs:
            if np.all(state_action_pair[0] == sorted_state) and np.all(state_action_pair[1] == sorted_action):
                return True
        # Otherwise, add it to the list
        self.state_action_pairs.append((sorted_state, sorted_action))
        return False

    def _add_to_state_reward_history(self,
                                     state: np.ndarray,
                                     reward: float,
                                     predicted_reward: Optional[float] = None) -> bool:
        # Add the (state, reward) pair to the cache and determines whether it is a new global optimum
        self.state_reward_history.append((state, reward, predicted_reward))
        if np.all(np.sort(state) == self.best_state()):
            return True
        return False

    def _write_to_log(self,
                      state: np.ndarray,
                      reward: float,
                      predicted_reward: Optional[float] = None,
                      is_best_state: bool = False,
                      step_counter: Optional[int] = None):
        step_counter = self.step_counter if step_counter is None else step_counter
        msg = f'{str(step_counter).rjust(3)} | {str(state)[1:-1].ljust(11)} | {reward:.5f}'
        msg = f'{msg} | pred={predicted_reward:.5f}' if predicted_reward is not None else f'{msg} |        '
        color = Colors.OKGREEN if is_best_state else None
        msg = f'{msg} | new best state' if is_best_state else msg
        self._write_message_to_log(msg, color)

    def _write_message_to_log(self, message: str, color: Optional[Colors] = None, add_to_log_history: bool = False):
        if add_to_log_history:
            self.log_history[self.step_counter] = (message, color)
        current_time = datetime.now().strftime('%H:%M:%S')
        msg = f'{current_time} | {message}'
        if color is not None:
            print(color + msg + Colors.ENDC)
        else:
            print(msg)
        with open(self.logfile, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
