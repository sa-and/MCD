from typing import Tuple, List, NoReturn, Union, Any
from abc import ABC, abstractmethod
from causalnex.structure import StructureModel
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations, permutations
import random
from gym.spaces import Discrete, Box


class CausalAgent(ABC):
    """
    The base class for all agents which maintain an epistemic causal graph about their environment.
    """
    var_names: Union[int, List[str]]
    causal_model: StructureModel
    collected_data: dict
    actions: List[Any]
    # state_repeats: int

    def __init__(self, vars: Union[int, List[str]],
                 causal_graph: StructureModel = None,
                 state_repeats: int = 1,
                 allow_interventions: bool = True):
        self.allow_interventions = allow_interventions
        if type(vars) == int:
            self.var_names = ['x' + str(i) for i in range(vars)]
        else:
            self.var_names = vars

        # initialize causal model
        if causal_graph:
            self.causal_model = causal_graph
        else:
            self.causal_model = StructureModel()
            [self.causal_model.add_node(name) for name in self.var_names]
            self.reset_causal_model()

        # initialize the storages for observational and interventional data.
        self.collected_data = {}

        self.action_space = None
        self.observation_space = None
        self.actions = []
        self.current_action = None
        self.state_repeats = state_repeats

    # --------------------------- Methods for maintaining the causal structure of the agent ---------------------------
    def set_causal_model(self, causal_model: StructureModel):
        self.causal_model = causal_model

    def reset_causal_model(self, mode: str = 'random'):
        """
        Sets the causal graph of the agent to either a graph with random edges or without edges at all.
        :param mode: 'random' or 'empty'
        """
        all_pairs = [(v[0], v[1]) for v in permutations(self.var_names, 2)]

        if mode == 'random':
            random.shuffle(all_pairs)
            for p in all_pairs:
                self.update_model(p, random.choice([0, 1, 2]))

        elif mode == 'empty':
            # delete all edges
            for p in all_pairs:
                self.update_model(p, 0)
        else:
            raise TypeError('No reset defined for mode ' + mode)

    def update_model(self, edge: Tuple[str, str],
                     manipulation: int,
                     allow_disconnecting: bool = True,
                     allow_cycles: bool = True) -> bool:
        """
        Updates model according to action and returns the success of the operation. Reversing and removing an edge that
        doesn't exists has no effect. Adding an edge which already exists has no effect.

        :param edge: The edge to be manipulated. e.g. (X0, X1)
        :param manipulation: 0 = remove edge, 1 = add edge, 2 = reverse edge
        :param allow_disconnecting: If true, manipulations which disconnect the causal graph can be executed.
        :param allow_cycles: If true, manipulations which result in a cycle can be executed.
        :return: True if the manipulation was successful. False if it wasn't or it was illegal according to
        'allow_disconnecting' or 'allow_cycles'.
        """

        if manipulation == 0:  # remove edge if exists
            if self.causal_model.has_edge(edge[0], edge[1]):
                self.causal_model.remove_edge(edge[0], edge[1])
                removed_edge = (edge[0], edge[1])
            else:
                return False

            # disconnected graph
            if not allow_disconnecting and nx.number_weakly_connected_components(self.causal_model) > 1:
                self.causal_model.add_edge(removed_edge[0], removed_edge[1])
                return False

        elif manipulation == 1:  # add edge
            if not self.causal_model.has_edge(edge[0], edge[1]):  # only add edge if not already there
                self.causal_model.add_edge(edge[0], edge[1])
            else:
                return False

            if not nx.is_directed_acyclic_graph(self.causal_model) and not allow_cycles:  # check if became cyclic
                self.causal_model.remove_edge(edge[0], edge[1])
                return False

        elif manipulation == 2:  # reverse edge
            if self.causal_model.has_edge(edge[0], edge[1]):
                self.causal_model.remove_edge(edge[0], edge[1])
                self.causal_model.add_edge(edge[1], edge[0])
                added_edge = (edge[1], edge[0])
            else:
                return False

            if not nx.is_directed_acyclic_graph(self.causal_model) and not allow_cycles:  # check if became cyclic
                self.causal_model.remove_edge(added_edge[0], added_edge[1])
                self.causal_model.add_edge(added_edge[1], added_edge[0])
                return False

        return True

    def display_causal_model(self) -> NoReturn:
        fig, ax = plt.subplots()
        nx.draw_circular(self.causal_model, ax=ax, with_labels=True)
        fig.show()

    def get_graph_state(self) -> List[float]:
        """
        Get a list of values that represents the state of an edge in the causal graph for each possible graph.
        The edges are ordered in lexographical order.

        Example:
        In a 3 node graph there are the potential edges: 0-1, 0-2, 1-2. The list [0, 0.5, 1] represents the
        graph 0x1, 0->2, 1<-2, where x means that there is no edge.

        :return: state of the graph
        """
        graph_state = []
        possible_edges = [e for e in combinations(self.var_names, 2)]
        for e in possible_edges:
            if self.causal_model.has_edge(e[0], e[1]):
                graph_state.append(0.5)
            elif self.causal_model.has_edge(e[1], e[0]):
                graph_state.append(1.0)
            else:
                graph_state.append(0.0)
        return graph_state

    def is_legal_intervention(self, interv_var: str) -> bool:
        """
        Checks if performing an intervention disconnects the graph. If it does, it is not a legal intervention
        for the causalnex library.
        :param interv_var: variable to intervene on.
        :return: False if an intervention on 'interv_var' would disconnect the graph.
        """
        model = self.causal_model.copy()
        nodes = nx.nodes(model)
        for n in nodes:
            if model.has_edge(n, interv_var):
                model.remove_edge(n, interv_var)
        is_connected = nx.number_weakly_connected_components(model) <= 1
        return is_connected

    # ---------------------------------------------- Abstract methods ----------------------------------------------
    @abstractmethod
    def get_action_from_actionspace_sample(self, sample: Any):
        raise NotImplementedError

    @abstractmethod
    def update_model_per_action(self, action: Any):
        raise NotImplementedError


class DiscreteAgent(CausalAgent):
    current_mode: str
    action_space: Discrete

    def __init__(self, n_vars: int,
                 causal_graph: StructureModel = None,
                 state_repeats: int = 1,
                 allow_interventions: bool = True):
        super(DiscreteAgent, self).__init__(n_vars, causal_graph, state_repeats=state_repeats,
                                            allow_interventions=allow_interventions)
        # create a list of actions that can be performed
        if self.allow_interventions:
            self.actions = [(0, i, 5.0) for i in range(n_vars)]

        # actions for graph manipulation represented as (1, edge, operation)
        # where operation can be one of: delete = 0, add = 1, reverse = 2
        edges = [e for e in combinations(self.var_names, 2)]
        edges.extend([(e[1], e[0]) for e in edges])
        for i in range(3):
            self.actions.extend([(1, edge, i) for edge in edges])
        self.actions.append((None, None, None))
        self.current_action = (None, None, None)
        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(-7.0, 7.0, (state_repeats * (int((n_vars * 2) + n_vars * (n_vars - 1) / 2)) + 1,))

    def update_model_per_action(self, action) -> bool:
        """Updates model according to action and returns the success of the operation"""
        assert action[0] == 1, "Action is not a b model manipulation."
        edge = action[1]
        manipulation = action[2]

        return self.update_model(edge, manipulation)

    def get_action_from_actionspace_sample(self, sample: int):
        return self.actions[sample]


