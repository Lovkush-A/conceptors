tasks=("antonyms" "capitalize" "country-capital" "english-french" "present-past" "singular-plural")
# compound_tasks=("english-french-antonym" "english-french-capitalized" "singular-plural-capitalized")

for task in "${tasks[@]}"
do
    echo "GPT-Neox 20B - addition - $task"
    sbatch -p g48 --output=logs/slurm-%j-$task-addition-neox.out run_experiment.sh \
        --experiment_type=addition \
        --model_name=EleutherAI/gpt-neox-20b \
        --task=$task \
        --results_path=../../results/ \
        --list_extraction_layers=1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31,33,34,35,37,38,39,41,42 \
        --list_beta_averaging=0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0
done
