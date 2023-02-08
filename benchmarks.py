import sys
import os
from typing import List, Tuple
from envs.generation.scm_gen import StructuralCausalModel
from envs.generation.graph_gen import CausalGraphGenerator
from abc import ABC, abstractmethod
from networkx import DiGraph
import argparse
import numpy as np
import torch
import copy
import networkx as nx
import csv
import shutil
from notears.linear import notears_linear
import time
sys.path.append(os.getcwd()+'\\benchmarking\\third_party\\ENCO')
from benchmarking.third_party.ENCO.causal_discovery.enco import ENCO
from benchmarking.third_party.ENCO.causal_graphs.graph_definition import CausalDAGDataset
from benchmarking.third_party.dcdi.dcdi.main import main as dcdimain


class BaseBenchmark(ABC):
    """
    Base class for benchmark algorithms to generate the causal structure of a given SCM. The classes implementing this
    class are meant to provide wrappers from our environments to existing Causal Discovery algorithms.
    """
    def __init__(self, evaluation_scm: StructuralCausalModel):
        """
        :param evaluation_scm: SCM on which to run the causal discovery algorithm.
        """
        self.evaluation_scm = evaluation_scm

    @abstractmethod
    def estimate_structure(self, n_obs_samples: int = 1000, n_int_samples_per_var: int = 300) -> DiGraph:
        """
        Estimate the causal structure of the current SCM environment.

        :param n_obs_samples: Amount of samples to take from the SCM in order to find its causal structure
        :param n_int_samples_per_var: Amount of samples to take for each variable.
        """
        raise NotImplementedError

    def sample_data(self, n_obs_samples: int = 1000, n_int_samples_per_var: int = 300) -> Tuple[List[list], dict]:
        # sample observational and interventional data
        obs_data = []
        int_data = {var: [] for var in self.evaluation_scm.endogenous_vars}
        for _ in range(n_obs_samples):  # observational case
            obs_data.append(self.evaluation_scm.get_next_instantiation()[0])
        for _ in range(n_int_samples_per_var):  # interventional case
            for target in self.evaluation_scm.endogenous_vars.keys():
                scm = copy.deepcopy(self.evaluation_scm)
                scm.do_interventions([(target, lambda: 5.0)])
                int_data[target].append(scm.get_next_instantiation()[0])
        return obs_data, int_data


class BenchmarkRandomCD(BaseBenchmark):
    """
    This class estimates the causal structure by creating a random DAG based on the given amount of variables
    """
    def __int__(self, evaluation_scm: StructuralCausalModel):
        """
        :param evaluation_scm: SCM on which to run the causal discovery algorithm.
        :param allow_confounding: Whether the generated DAG should have exogenous confounders (True) or not (False).
        :param seed: Seed for random number generator.
        """
        super(BenchmarkRandomCD, self).__init__(evaluation_scm)

    def estimate_structure(self, n_obs_samples: int = 1000, n_int_samples_per_var: int = 300) -> DiGraph:
        gen = CausalGraphGenerator(len(self.evaluation_scm.endogenous_vars),
                                   len(self.evaluation_scm.exogenous_vars))
        start = time.perf_counter()
        graph = gen.generate_random_graph()[0]
        end = time.perf_counter()
        return graph, end-start


class BenchmarkENCO(BaseBenchmark):
    """
    Class to apply ENCO (Lippe et al., 2022) to our environments.
    """
    def __init__(self, evaluation_scm: StructuralCausalModel):
        super(BenchmarkENCO, self).__init__(evaluation_scm)

    def estimate_structure(self, n_obs_samples: int = 1000, n_int_samples_per_var: int = 300) -> [DiGraph, float]:
        # sample observational and interventional data
        obs_data, int_data = self.sample_data(n_obs_samples, n_int_samples_per_var)
        # Put the data in the right format
        obs_data = np.array(obs_data)
        int_data = np.dstack(int_data.values()).swapaxes(0, 1)

        # Create dag to learn on
        graph = CausalDAGDataset(adj_matrix=np.zeros((len(self.evaluation_scm.endogenous_vars),
                                                      len(self.evaluation_scm.endogenous_vars))),
                                 data_obs=obs_data,
                                 data_int=int_data)

        # let the bach-size never be bigger than the sample size to avoid bugs
        if n_obs_samples < 128 or n_int_samples_per_var < 128:
            model = ENCO(graph,
                         sample_size_obs=len(obs_data),
                         sample_size_inters=len(int_data),
                         batch_size=min(n_obs_samples, n_int_samples_per_var))
        else:
            model = ENCO(graph, sample_size_obs=len(obs_data), sample_size_inters=len(int_data))

        if torch.cuda.is_available():
            model.to(torch.device('cuda:0'))
        # apply ENCO to the data
        start = time.perf_counter()
        predicted_adj_matrix = model.discover_graph(num_epochs=50)
        end = time.perf_counter()

        # convert adj_matrix to DiGraph
        graph = nx.from_numpy_matrix(np.array(predicted_adj_matrix.double()), create_using=nx.DiGraph)
        # rename nodes to fit the schema
        nx.relabel_nodes(graph, {i:'X'+str(i) for i in graph.nodes}, copy=False)
        return graph, end-start


class BenchmarkDCDI(BaseBenchmark):
    def __init__(self, evaluation_scm: StructuralCausalModel):
        super(BenchmarkDCDI, self).__init__(evaluation_scm)

    def estimate_structure(self, n_obs_samples: int = 1000, n_int_samples_per_var: int = 300) -> DiGraph:
        n_vars = len(self.evaluation_scm.endogenous_vars)
        # sample observational and interventional data
        obs_data, int_data = self.sample_data(n_obs_samples, n_int_samples_per_var)
        obs_data = np.array(obs_data)

        # reformatting interventional data
        int_data = np.array([np.array(int_data[target]).flatten() for target in int_data.keys()]).flatten()\
            .reshape((n_int_samples_per_var*len(self.evaluation_scm.endogenous_vars), n_vars))

        # generating the right files. Code adapted from dcdi/data/generation/generate_data.py generate()
        regimes = []
        regimes.extend([0 for _ in range(n_obs_samples)])
        [regimes.extend([i+1 for _ in range(n_int_samples_per_var)]) for i in range(n_vars)]
        regimes.reverse()
        regimes = np.array(regimes)

        mask_intervention = []
        [mask_intervention.extend([i for _ in range(n_int_samples_per_var)]) for i in range(n_vars)]
        mask_intervention = np.array(mask_intervention)

        # get ground truth adj. This is needed for metrics and plotting
        adj = nx.adjacency_matrix(self.evaluation_scm.create_graph()).todense()

        # save data into the desired format
        os.mkdir('./temp/')
        try:
            np.save('./temp/data_interv1.npy', int_data)#np.concatenate((int_data,obs_data)))
            with open("./temp/intervention1.csv", 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows([[i] for i in mask_intervention.tolist()])
            with open("./temp/regime1.csv", 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows([[i] for i in regimes.tolist()])
            np.save("./temp/DAG1.npy", adj)

            # bring the arguments in the right format
            parser = self._get_parser()
            args = parser.parse_args(['--train', '--num-vars', str(n_vars), '--model', 'DCDI-DSF', '--exp-path', './temp/',
                                      '--data-path', './temp/', '--i-dataset', '1', '--intervention',
                                      '--num-train-iter', '50000', '--plot-freq', '100000000',
                                      '--train-batch-size', str(min(64, len(int_data)))])

            # run the DCDI training algorithm
            start = time.perf_counter()
            model = dcdimain(args)
            end = time.perf_counter()
            # consider threshold
            thresh_adj = (model.get_w_adj() > 0.5).type(torch.Tensor)
            graph = nx.from_numpy_matrix(thresh_adj.detach().numpy(), create_using=nx.DiGraph)
            # rename nodes to fit the schema
            nx.relabel_nodes(graph, {i: 'X' + str(i) for i in graph.nodes}, copy=False)
            return graph, end-start

        finally:
            shutil.rmtree('./temp')

    def _get_parser(self):
        """
        Returns the parser which provides the right arguments exactly following the dcdi/main script
        """
        parser = argparse.ArgumentParser()

        # experiment
        parser.add_argument('--exp-path', type=str, default='/exp',
                            help='Path to experiments')
        parser.add_argument('--train', action="store_true",
                            help='Run `train` function, get /train folder')
        parser.add_argument('--retrain', action="store_true",
                            help='after to-dag or pruning, retrain model from scratch before reporting nll-val')
        parser.add_argument('--dag-for-retrain', default=None, type=str, help='path to a DAG in .npy \
                                format which will be used for retrainig. e.g.  /code/stuff/DAG.npy')
        parser.add_argument('--random-seed', type=int, default=42, help="Random seed for pytorch and numpy")

        # data
        parser.add_argument('--data-path', type=str, default=None,
                            help='Path to data files')
        parser.add_argument('--i-dataset', type=str, default=None,
                            help='dataset index')
        parser.add_argument('--num-vars', required=True, type=int, default=2,
                            help='Number of variables')
        parser.add_argument('--train-samples', type=int, default=0.8,
                            help='Number of samples used for training (default is 80% of the total size)')
        parser.add_argument('--test-samples', type=int, default=None,
                            help='Number of samples used for testing (default is whatever is not used for training)')
        parser.add_argument('--num-folds', type=int, default=5,
                            help='number of folds for cross-validation')
        parser.add_argument('--fold', type=int, default=0,
                            help='fold we should use for testing')
        parser.add_argument('--train-batch-size', type=int, default=64,
                            help='number of samples in a minibatch')
        parser.add_argument('--num-train-iter', type=int, default=1000000,
                            help='number of meta gradient steps')
        parser.add_argument('--normalize-data', action="store_true",
                            help='(x - mu) / std')
        parser.add_argument('--regimes-to-ignore', nargs="+", type=int,
                            help='When loading data, will remove some regimes from data set')
        parser.add_argument('--test-on-new-regimes', action="store_true",
                            help='When using --regimes-to-ignore, we evaluate performance on new regimes never seen during'
                                 ' training (use after retraining).')

        # model
        parser.add_argument('--model', type=str, required=True,
                            help='model class (DCDI-G or DCDI-DSF)')
        parser.add_argument('--num-layers', type=int, default=2,
                            help="number of hidden layers")
        parser.add_argument('--hid-dim', type=int, default=16,
                            help="number of hidden units per layer")
        parser.add_argument('--nonlin', type=str, default='leaky-relu',
                            help="leaky-relu | sigmoid")
        parser.add_argument("--flow-num-layers", type=int, default=2,
                            help='number of hidden layers of the DSF')
        parser.add_argument("--flow-hid-dim", type=int, default=16,
                            help='number of hidden units of the DSF')

        # intervention
        parser.add_argument('--intervention', action="store_true",
                            help="Use data with intervention")
        parser.add_argument('--dcd', action="store_true",
                            help="Use DCD (DCDI with a loss not taking into account the intervention)")
        parser.add_argument('--intervention-type', type=str, default="perfect",
                            help="Type of intervention: perfect or imperfect")
        parser.add_argument('--intervention-knowledge', type=str, default="known",
                            help="If the targets of the intervention are known or unknown")
        parser.add_argument('--coeff-interv-sparsity', type=float, default=1e-8,
                            help="Coefficient of the regularisation in the unknown \
                                interventions case (lambda_R)")

        # optimization
        parser.add_argument('--optimizer', type=str, default="rmsprop",
                            help='sgd|rmsprop')
        parser.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate for optim')
        parser.add_argument('--lr-reinit', type=float, default=None,
                            help='Learning rate for optim after first subproblem. Default mode reuses --lr.')
        parser.add_argument('--lr-schedule', type=str, default=None,
                            help='Learning rate for optim, change initial lr as a function of mu: None|sqrt-mu|log-mu')
        parser.add_argument('--stop-crit-win', type=int, default=100,
                            help='window size to compute stopping criterion')
        parser.add_argument('--reg-coeff', type=float, default=0.1,
                            help='regularization coefficient (lambda)')

        # Augmented Lagrangian options
        parser.add_argument('--omega-gamma', type=float, default=1e-4,
                            help='Precision to declare convergence of subproblems')
        parser.add_argument('--omega-mu', type=float, default=0.9,
                            help='After subproblem solved, h should have reduced by this ratio')
        parser.add_argument('--mu-init', type=float, default=1e-8,
                            help='initial value of mu')
        parser.add_argument('--mu-mult-factor', type=float, default=2,
                            help="Multiply mu by this amount when constraint not sufficiently decreasing")
        parser.add_argument('--gamma-init', type=float, default=0.,
                            help='initial value of gamma')
        parser.add_argument('--h-threshold', type=float, default=1e-8,
                            help='Stop when |h|<X. Zero means stop AL procedure only when h==0')

        # misc
        parser.add_argument('--patience', type=int, default=10,
                            help='Early stopping patience in --retrain.')
        parser.add_argument('--train-patience', type=int, default=5,
                            help='Early stopping patience in --train after constraint')
        parser.add_argument('--train-patience-post', type=int, default=5,
                            help='Early stopping patience in --train after threshold')

        # logging
        parser.add_argument('--plot-freq', type=int, default=10000,
                            help='plotting frequency')
        parser.add_argument('--no-w-adjs-log', action="store_true",
                            help='do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)')
        parser.add_argument('--plot-density', action="store_true",
                            help='Plot density (only implemented for 2 vars)')

        # device and numerical precision
        parser.add_argument('--gpu', action="store_true",
                            help="Use GPU")
        parser.add_argument('--float', action="store_true",
                            help="Use Float precision")

        return parser


class BenchmarkNOTEARS(BaseBenchmark):
    def __init__(self, evaluation_scm: StructuralCausalModel):
        super(BenchmarkNOTEARS, self).__init__(evaluation_scm)

    def estimate_structure(self, n_obs_samples: int = 1000, n_int_samples_per_var: int = 300) -> DiGraph:
        # collect data from current env
        obs_data, _ = self.sample_data(n_obs_samples, n_int_samples_per_var=0)

        # estimate weights
        obs_data = np.array(obs_data)
        start = time.perf_counter()
        adj_matrix = notears_linear(obs_data, lambda1=0.01, loss_type='l2', w_threshold=0.1)
        end = time.perf_counter()

        # build digraph form weight matrix
        predicted_graph = nx.DiGraph()
        [predicted_graph.add_node('X'+str(i)) for i in range(len(self.evaluation_scm.endogenous_vars))]
        for row in range(len(adj_matrix)):
            for col in range(len(adj_matrix)):
                if adj_matrix[row][col] > 0.0:
                    predicted_graph.add_edge('X'+str(row), 'X'+str(col))

        return predicted_graph, end-start