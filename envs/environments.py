"""Defines the classes for the Gym environments that use an SCM"""

from typing import List, Callable, Tuple, NoReturn, Optional
import networkx as nx
from gym import Env
from agents import CausalAgent, DiscreteAgent
import copy
import numpy as np
from envs.generation.scm_gen import SCMGenerator, StructuralCausalModel
from envs.generation.graph_gen import CausalGraphGenerator
from episode_evals import EvalFunc, FixedLengthStructEpisode


class SCMEnvironment(Env):
    Agent: DiscreteAgent
    Function = Callable[[], bool]
    Lights: List[bool]

    def __init__(self, agent: CausalAgent,
                 eval_func: EvalFunc,
                 episode_length: int,
                 scm: StructuralCausalModel):
        super(SCMEnvironment, self).__init__()
        self.metrics = {'ep_lengths': [],
                        'rewards': []}
        self.eval_func = eval_func
        self.episode_length = episode_length

        # initialize causal model
        self.scm = scm
        self.var_values = self.scm.get_next_instantiation()[0]
        if type(eval_func) == FixedLengthStructEpisode:
            eval_func.set_compare_graph(self.scm.create_graph())

        self.agent = agent
        self.action_space = self.agent.action_space
        self.observation_space = self.agent.observation_space
        self.prev_action = None
        self.old_obs = []

        self.steps_this_episode = 0
        self.observation = None
        self.update_obs_vector()

    def reset(self) -> np.ndarray:
        self.steps_this_episode = 0
        self.agent.reset_causal_model(mode='random')
        # reset observations
        self.old_obs = []
        self.update_obs_vector()
        return self.observation

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        self.agent.current_action = self.agent.get_action_from_actionspace_sample(action)

        # apply action
        curr_scm = self.scm
        action_successful = False
        if self.agent.current_action[0] == 0:  # intervention action
            curr_scm = copy.deepcopy(self.scm)
            curr_scm.do_interventions([('X'+str(self.agent.current_action[1]), lambda: self.agent.current_action[2])])
            action_successful = True
        elif self.agent.current_action[0] == 1:
            action_successful = self.agent.update_model_per_action(self.agent.current_action)
        elif self.agent.current_action[0] == None or self.agent.current_action[0] == -1:
            action_successful = True

        self.steps_this_episode += 1

        # determine the states of the lights according to the causal structure
        self.var_values = curr_scm.get_next_instantiation()[0]

        # determine state after action
        self.update_obs_vector()

        # evaluate the step
        done, reward = self.eval_func.evaluate_step(action_successful)
        # give small bonus for interventions
        if self.agent.current_action[0] == 0:
            reward += 0.1

        self.prev_action = self.agent.current_action
        self.metrics['rewards'].append(reward)

        # reset environment if episode is done
        if done:
            self.reset()

        return self.observation, reward, done, {}

    def update_obs_vector(self):
        intervention_one_hot = [1.0 if self.agent.current_action[1] == i else 0.0 for i in range(len(self.var_values))]
        graph_state = self.agent.get_graph_state()
        steps_left = (self.episode_length - self.steps_this_episode) / self.episode_length
        state = [float(l) for l in self.var_values]  # convert bool to float
        state.extend(intervention_one_hot)
        state.extend(graph_state)
        state.extend([steps_left])
        self.old_obs = state
        self.observation = np.array(self.old_obs).flatten()

    def render(self, mode: str = 'human') -> NoReturn:
        if mode == 'human':
            out = ''
            for i in range(len(self.var_values)):
                if self.var_values[i]:
                    out += '|'
                else:
                    out += 'O'
                if self.agent.current_action[1] == i:
                    out += '*'
                out += '\t'
            print(out)


class SCMEnvironmentReservoir(Env):
    envs: List[SCMEnvironment]

    def __init__(self, n_vars: int,
                 agent_type: type(CausalAgent),
                 eval_func_type: type(EvalFunc),
                 episode_length: int,
                 possible_functions: List[str],
                 test_set: Optional[List[nx.DiGraph]] = None):

        self.envs = []
        self.test_set = test_set
        self.eval_func_type = eval_func_type
        self.agent_type = agent_type
        self.episode_length = episode_length
        self.n_vars = n_vars
        self.possible_functions = possible_functions
        self.gen = SCMGenerator()

        self.current_env = self.get_next_env()
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def _build_agent_eval_func(self):
        agent = self.agent_type(self.n_vars)
        rand_graph = CausalGraphGenerator(self.n_vars, 0).generate_random_graph()[0]
        eval_func = self.eval_func_type(agent, graph=rand_graph, ep_length=self.episode_length)
        return agent, eval_func

    def reset(self):
        # reset the current environment
        self.current_env.reset()

        # choose a random next environment and reset it
        self.current_env = self.get_next_env()
        return self.current_env.reset()

    def step(self, action):
        return self.current_env.step(action)

    def render(self, mode='human'):
        self.current_env.render(mode)

    def get_next_env(self):
        # resample scm until there is one that is not in the test set
        while True:
            scm = self.gen.create_random(possible_functions=[k for k in self.possible_functions],
                                         n_endo=self.n_vars, n_exo=0)[0]
            graph = scm.create_graph()

            in_testset = False
            for i in range(len(self.test_set)):
                edit_distance = nx.graph_edit_distance(self.test_set[i], graph)
                if edit_distance == 0:
                    in_testset = True
                    break
            if not in_testset:
                break
        agent, eval_func = self._build_agent_eval_func()
        return SCMEnvironment(agent, eval_func, self.episode_length, scm)

