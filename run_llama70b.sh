sbatch -p g48 run_experiment.sh \
    --experiment_type=conceptor --model_name=meta-llama/Llama-2-70b-chat-hf \
    --list_extraction_layers=0,16,32