"""Training script to train an MCD policy"""

import argparse
import copy
import dill
from envs.environments import SCMEnvironmentReservoir
from envs.callbacks import EvalTrainTestCallback
from agents import DiscreteAgent
from episode_evals import FixedLengthStructEpisode
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines import ACER
import stable_baselines.common.vec_env as venv
from stable_baselines.common.callbacks import CheckpointCallback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, help='Filepath of where to save the data.')
    parser.add_argument('--total-steps', type=int, help='Total amount of steps to train the model.')
    parser.add_argument('--ep-length', type=int, default=20, help='Episode length.')
    parser.add_argument('--n-vars', type=int, default=3, help='Number of endogenous variables.')
    parser.add_argument('--test-set', type=str, help='Path to pickled file with testing data.')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers to work in parallel.')
    parser.add_argument('--val-frequency', type=int, default=200000,
                        help='Frequency in training steps in which the agent should be evaluated.')
    parser.add_argument('--load-model-path', default=None, type=str, help='Path to load a pretrained model from.'
                                                                    ' \'None\' if it should be trained from scratch.')
    parser.add_argument('--n-eval-episodes', type=int, default=200, help='How many episodes should be done for '
                                                                         'each evaluation.')

    args = parser.parse_args()

    # load data
    with open(args.test_set+'graphs.pkl', "rb") as f:
        test_dags = dill.load(f)
    with open(args.test_set+'scms.pkl', "rb") as f:
        test_scms = dill.load(f)

    # create the environment
    env = SCMEnvironmentReservoir(args.n_vars, DiscreteAgent, FixedLengthStructEpisode,
                                  args.ep_length, possible_functions=['linear'], test_set=test_dags)

    # create vectorized environments for parallelization
    env = venv.SubprocVecEnv([lambda: copy.deepcopy(env) for _ in range(args.workers)], start_method='spawn')

    # load pretrained model is specified
    if args.load_model_path:
        model = ACER.load(args.load_model_path, env, **{'buffer_size': 500000})

    # Create new model if not specified
    else:
        model = ACER(MlpLstmPolicy, env,
                     policy_kwargs={'net_arch': [30,
                                                 'lstm',
                                                 {'pi': [30],
                                                  'vf': [10]}],
                                    'n_lstm': 30},
                     buffer_size=500000,
                     lr_schedule="constant",
                     n_cpu_tf_sess=args.workers
                     )

    # set initial values for training
    steps = 0
    n_worse_than_best = 0
    best_shd = 10000000
    best_model = None
    val_shds = []
    train_shds = []

    # setup callbacks
    checkpoint_cb = CheckpointCallback(save_freq=int(args.val_frequency/args.workers),
                                       save_path=args.save_dir,
                                       name_prefix='latest_model')
    eval_cb = EvalTrainTestCallback(val_frequency=int(args.val_frequency/args.workers),
                                    scms_val=test_scms,
                                    n_vars=args.n_vars,
                                    episode_length=args.ep_length,
                                    best_model_save_path=args.save_dir,
                                    log_path=args.save_dir)

    # main training loop
    model.learn(args.total_steps, callback=[checkpoint_cb, eval_cb])
