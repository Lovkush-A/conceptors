#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH -t 24:00:00

# Check if the virtual environment exists
if [ ! -d "venv" ]; then
  echo "Virtual environment not found! Please ensure you have created it in the 'venv' directory."
  exit 1
fi

# Activate the virtual environment
source venv/bin/activate

export HF_HOME=/export/work/sabreu/.cache/

# Run the Python script with all command line arguments
cd src/functionvectors
python -u run_experiment.py "$@"
