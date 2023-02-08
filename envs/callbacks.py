"""For defining eventual callback functions for the training loop"""

from stable_baselines.common.callbacks import EventCallback, BaseCallback
from evaluation import evaluate_policy
import os
import numpy as np
from typing import Optional


class EvalTrainTestCallback(EventCallback):
    """Slightly adapted version of stable_baselines.common.callbacks.EvalCallback"""
    def __init__(self, val_frequency: int, scms_val, n_vars: int, episode_length: int,
                 best_model_save_path: str, log_path: str, verbose: int = 1,
                 callback_on_new_best: Optional[BaseCallback] = None) -> None:
        super(EvalTrainTestCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.val_frequency = val_frequency
        self.scms_val = scms_val
        self.total_steps = 0
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.n_vars = n_vars
        self.episode_length = episode_length
        self.evaluations_timesteps = []
        self.evaluations_results_val = []
        self.best_mean_reward = np.inf

    def _init_callback(self) -> None:
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.val_frequency > 0 and self.n_calls % self.val_frequency == 0:
            episode_rewards_val = evaluate_policy(self.model, self.scms_val, 1, self.n_vars,
                                                  self.episode_length, display=False, printing=False).mean()

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results_val.append(episode_rewards_val)
                np.savez(self.log_path+'/log',
                         timesteps=self.evaluations_timesteps,
                         results_val=self.evaluations_results_val)

            mean_reward_val = np.mean(episode_rewards_val)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward_val = mean_reward_val

            if self.verbose > 0:
                print("Eval val num_timesteps={}, "
                      "avg_shd={:.2f}".format(self.num_timesteps, mean_reward_val))

            if mean_reward_val < self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, 'best_val_model'))
                self.best_mean_reward = mean_reward_val

                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
            print()

        return True

