# Environment Setup

## Table of Contents
- [1. ALFWorld](#1-alfworld)  
- [2. WebShop](#2-webshop)  
- [3. Sokoban](#3-sokoban)  
- [4. Gym Cards](#4-gym-cards)  
- [5. AppWorld (Experimental)](#5-appworld-experimental)  

## 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
pip install vllm==0.8.5
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

## 2. WebShop
Use a dedicated Python 3.10 environment for the text simulator and training stack.
From the repository root on a Linux GPU server:

```bash
python -m recipe.denoise_v2.task_suite.setup_environment --benchmark webshop --mode fresh
conda activate denoise-webshop
python -m recipe.denoise_v2.task_suite.prepare_webshop_assets --download --build-index
```

The old `webshop/setup.sh` changes the active environment and is disabled by default.
See [isolated environments](../../recipe/denoise_v2/task_suite/ENVIRONMENTS.md) for
cloning a healthy existing environment, ScienceWorld installation, task manifests,
and simulator / training checks. Both GRPO and DenoiseRL use the same environment
within each benchmark. Resolve dependency conflicts before training.

---
## 3. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```
---
## 4. Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```
---
### 5. AppWorld (Experimental)
Install AppWorld package
```bash
cd repo_root/
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
pip install -e .
pip install vllm==0.8.5
```
You can ignore the warning of incompatiblity for appworld, because we don't run appworld in `verl-agent` environment.

Create a dedicated conda environment `appworld` for the AppWorld server:
```bash
conda create -n appworld python=3.12 -y
conda activate appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
```
