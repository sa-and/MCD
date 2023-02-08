from benchmarks import BenchmarkENCO, BenchmarkRandomCD, BenchmarkDCDI, BenchmarkNOTEARS
import argparse
import dill
from evaluation import directed_shd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-data', type=str, help='Filepath to the data to evaluate.')
    parser.add_argument('--benchmarks', nargs='+', default=["ENCO"], help='The list of benchmarks to run. Must be subset'
                                                                          ' of ["ENCO", "random", "NOTEARS", "DCDI"]')
    parser.add_argument('--save-path', type=str, help='Directory to save the results.')
    parser.add_argument('--n-obs-samples', type=int, default=1000, help='How many observational data points to collect.')
    parser.add_argument('--n-int-samples', type=int, default=333, help='How many interventional data points to collect.')
    parser.add_argument('--max-n-scms', type=int, default=70, help='Maximum number of SCMs to evaluate')

    args = parser.parse_args()
    with open(args.val_data, "rb+") as f:
        scm_data = dill.load(f)

    for benchmark in args.benchmarks:
        results = []
        for scm in scm_data[:args.max_n_scms]:
            current_gt_graph = scm.create_graph()
            if benchmark == 'ENCO':
                bench = BenchmarkENCO(evaluation_scm=scm)
            elif benchmark == 'random':
                bench = BenchmarkRandomCD(evaluation_scm=scm)
            elif benchmark == 'DCDI':
                bench = BenchmarkDCDI(evaluation_scm=scm)
            elif benchmark == 'NOTEARS':
                bench = BenchmarkNOTEARS(evaluation_scm=scm)

            graph, time = bench.estimate_structure(n_obs_samples=args.n_obs_samples, n_int_samples_per_var=args.n_int_samples)
            shd = directed_shd(graph, current_gt_graph)
            results.append((shd, time))
        results = np.array(results)
        np.save(args.save_path, results)
