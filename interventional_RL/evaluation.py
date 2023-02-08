"""Script for running and evaluating a MCD policy"""

from stable_baselines import ACER
from envs.environments import SCMEnvironment
from envs.generation.scm_gen import SCMGenerator, StructuralCausalModel
from agents import DiscreteAgent
from episode_evals import NoEval
import networkx as nx
import numpy as np
from episode_evals import directed_shd
from typing import Union
from time import perf_counter


def apply_policy(model, test_env: Union[StructuralCausalModel, SCMEnvironment], n_vars, episode_length, display,
                 printing=True, wrapped_env=False):
    model_workers = model.n_envs
    if type(test_env) == StructuralCausalModel:
        test_env = SCMEnvironment(agent=DiscreteAgent(n_vars),
                                  scm=test_env,
                                  eval_func=NoEval(),
                                  episode_length=episode_length)

    # just do this multiple times for easier inspection
    states = model.initial_state
    done = [False for _ in range(model_workers)]
    obs = test_env.reset()
    obs = [obs for _ in range(model_workers)]

    for i in range(episode_length):
        if printing:
            pass #print(obs)
        actions, states = model.predict(obs, state=states, mask=done, deterministic=True)
        if printing:
            try:
                if wrapped_env:
                    agent_action = test_env.agent.actions.index(test_env.actions[actions[0]])
                else:
                    agent_action = actions[0]
            except ValueError:  # could not find the action
                agent_action = -1  # none action
            print(test_env.agent.get_action_from_actionspace_sample(agent_action))
        obs, _, done, _ = test_env.step(actions[0])
        obs = [obs for _ in range(model_workers)]
        done = [done for _ in range(model_workers)]
        if printing:
            test_env.render()
    if display:
        test_env.agent.display_causal_model()
    if printing:
        print('\n\n\n\n')
    return nx.DiGraph(test_env.agent.causal_model)


def evaluate_policy(model, eval_data, runs_per_env: int, n_vars: int,
                    episode_length: int, display: bool, printing: bool, wrapped_env: bool = False) -> np.array:
    if type(model) == str:
        model = ACER.load(model)

    results = []
    for scm in eval_data:
        target_graph = scm.create_graph()

        for run in range(runs_per_env):
            start = perf_counter()
            predicted_graph = apply_policy(model=model,
                                             test_env=scm,
                                             n_vars=n_vars,
                                             episode_length=episode_length,
                                             display=display,
                                             printing=printing,
                                             wrapped_env=wrapped_env)
            end = perf_counter()

            difference = directed_shd(predicted_graph, target_graph)
            results.append((difference, end-start))

    results = np.array(results)
    return results


if __name__ == '__main__':
    path = 'experiments/delme/'
    model = 'latest_model_10000_steps.zip'
    eval_data = SCMGenerator.load_dataset('data/3en_0ex_8g_lin/scms.pkl')[:50]
    runs = 1
    vars = 3

    diffs = []
    diffs = evaluate_policy(model=path+model, eval_data=eval_data,
                            runs_per_env=runs, n_vars=vars, episode_length=20,
                            display=False, printing=True, wrapped_env=False)

    print('mean:', diffs.mean())
    print('std:', diffs.std())

    np.save(path+'delme', diffs)

