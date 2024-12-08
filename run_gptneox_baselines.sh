tasks=("antonyms" "capitalize" "country-capital" "english-french" "present-past" "singular-plural")
# compound_tasks=("english-french-antonym" "english-french-capitalized" "singular-plural-capitalized")

for task in "${tasks[@]}"
do
    echo "GPT-Neox 20B - baseline - $task"
    sbatch -p g48 --output=logs/slurm-%j-$task-baseline-neox.out run_experiment.sh \
        --experiment_type=baseline \
        --model_name=EleutherAI/gpt-neox-20b \
        --task=$task \
        --results_path=../../results/
done
