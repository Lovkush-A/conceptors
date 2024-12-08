tasks=("antonyms" "capitalize" "country-capital" "english-french" "present-past" "singular-plural")
# compound_tasks=("english-french-antonym" "english-french-capitalized" "singular-plural-capitalized")

for task in "${tasks[@]}"
do
    echo "GPT-J 6B - addition - $task"
    sbatch -p g24 --output=logs/slurm-%j-$task-addition-gptj.out run_experiment.sh \
        --experiment_type=addition \
        --model_name=EleutherAI/gpt-j-6b \
        --task=$task \
        --results_path=../../results/ \
        --list_extraction_layers=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
        --list_beta_averaging=0.5,1.0,2.0,3.0,4.0,5.0
done
