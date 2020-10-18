#!/bin/bash
#SBATCH --account=def-jpineau
#SBATCH --gres=gpu:1
#SBATCH --mem=256M
#SBATCH --time=00-00:30

source ../../venv/bin/activate
python run_random_agent_pytorch.py
