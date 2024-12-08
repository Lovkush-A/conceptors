tasks=("antonyms" "capitalize" "country-capital" "english-french" "present-past" "singular-plural")
# compound_tasks=("english-french-antonym" "english-french-capitalized" "singular-plural-capitalized")

for task in "${tasks[@]}"
do
    echo "GPT-J 6B - baseline - $task"
    sbatch -p g24 --output=logs/slurm-%j-$task-baseline-gptj.out run_experiment.sh \
        --experiment_type=baseline \
        --model_name=EleutherAI/gpt-j-6b \
        --task=$task \
        --results_path=../../results/
done
