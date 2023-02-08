# A Meta-Reinforcement Learning Algorithm for Causal Discovery

## Getting Started
- Clone this repository.
- Download anaconda at https://www.anaconda.com/products/individual
- Create a new environment with python=3.7.0. You can do that from the terminal with  `conda create --name [name] python=3.7.0)`.
- Activate the new environment with `conda activate [name]`.
- Install all other packages with `pip install requirements.txt`
- If you are planning to use the benchmarks, make install/clone the corresponding code into the "benchmarking/third_party" folder. For getting DCDI to work you need to go to its project files and add `return model` in the last line of `dcdi.main.dcdimain`.

## Training the model
If you want to reproduce the training of our model, run the following for the 3-variable environments:
`python training.py --test-set data/3en_0ex_8g_lin/ --n-vars 3 --save-dir experiments/delme --total-steps 200000000`

And the following for the 4-variable environments:
`python training.py --test-set data/4en_0ex_200g_lin/ --n-vars 4 --save-dir experiments/delme --total-steps 200000000`

Make sure to also check the other parameters in train.py if you want more flexibility.

## Running benchmarks
To run the benchmarks, run the following:
`python run_benchmarks.py --val-data data/3en_0ex_8g_lin/scms.pkl --benchmarks BENCH --save-path experiments/test`
Where BENCH is one of ["ENCO", "random", "NOTEARS", "DCDI"]. See parameters in `run_benchmarks.py` for more flexibility.

## Generating the Graph Data and corresponding SCMs
If you want to generate environments yourself, try the following: Example for generating 1000 random DAGs with 5 endogenous and 0 exogenous variables and for each of these graphs 10 SCMs
with random linear functional relations.
``
python gen_data.py --n-graphs 1000 --scms-per-graph 10 --save-dir PATH\5en_0ex_1000g\ --n-endo 5 --n-exo 0 --seed 1
``

## Referencing
This code implements the algorithm of "A Meta-Reinforcement Learning Algorithm for Causal Discovery" by Andreas Sauter, Erman Acar, Vincent François-Lavet, 2022. If you are using this code, please reference
```
@misc{sauter22meta,
  doi = {10.48550/ARXIV.2207.08457},  
  url = {https://arxiv.org/abs/2207.08457},  
  author = {Sauter, Andreas and Acar, Erman and François-Lavet, Vincent},    
  title = {A Meta-Reinforcement Learning Algorithm for Causal Discovery},  
  publisher = {arXiv},  
  year = {2022},  
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```

