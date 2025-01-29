**Disclaimer**: this code has not been exhaustively tested. In case of issues, please start an issue on GitHub or contact the authors.

**Project Page**: https://sites.google.com/view/rmbpo

# Info
This code contains evaluation scripts and trained (policy) weights corresponding with the paper *Robust Reinforcement Learning in a Sample-Efficient Setting*. Additionally, the distorted environments are provided. The code relies on the [Unstable Baselines library](https://github.com/x35f/model_based_rl/tree/2c2c4aca4952e3251fd77ba9413b72d25c033426).


# How to install

 Install the conda environment and Unstable Baselines with the commands below:

```bash
conda env create -f env.yaml 
conda activate rmbpo
git clone --recurse-submodules https://github.com/x35f/unstable_baselines.git ./embodied/unstable_baselines/
pip install -e ./embodied/unstable_baselines
pip install -e ./embodied/robustness-gymnasium
pip install -e .
pip uninstall mujoco-py # We only want new mujoco bindings, mujoco-py might error on your system
```

# How to run evaluation
Use the following structure:

```bash
python evaluate/main_eval.py {PATH_TO_CONFIG} --load-dir {PATH TO_RMBPO_WEIGHTS} --num-eval-runs {NUM OF DESIRED RUNS PER SEED}
```

E.g.

```bash
python evaluate/main_eval.py configs/robust_mbpo/hopper-torso.py --load-dir ./weights/hopper-v4/rmbpo_4/
```
This will display a plot and save the evaluation results under *results_data/* as a .pkl file.

You can view all arguments via:
```bash
python evaluate/main_eval.py configs/hopper-mass.py --help
```

**Note:** the default evaluation length is 10 episodes, increase this to 30 to reproduce the paper.

# Toy Experiment

You can find the code for the toy experiment under *toy_experiment/toy_experiment.py* .
