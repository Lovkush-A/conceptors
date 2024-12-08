tasks=("antonyms" "capitalize" "country-capital" "english-french" "present-past" "singular-plural")
# compound_tasks=("english-french-antonym" "english-french-capitalized" "singular-plural-capitalized")

for task in "${tasks[@]}"
do
    echo "GPT-Neox 20B - conceptor - $task"
    sbatch -p g48 --output=logs/slurm-%j-$task-conceptor-neox.out run_experiment.sh \
        --experiment_type=conceptor \
        --model_name=EleutherAI/gpt-neox-20b \
        --task=$task \
        --results_path=../../results/ \
        --list_extraction_layers=0,4,8,12,16,20,24 \
        --list_beta_conceptor=0.5,1.0,2.0,3.0,4.0,5.0 \
        --list_apertures_normal=0.001,0.0125,0.05,0.1
done