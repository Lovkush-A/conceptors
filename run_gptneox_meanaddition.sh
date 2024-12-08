tasks=("antonyms" "capitalize" "country-capital" "english-french" "present-past" "singular-plural")
# compound_tasks=("english-french-antonym" "english-french-capitalized" "singular-plural-capitalized")

for task in "${tasks[@]}"
do
    echo "GPT-Neox 20B - addition-mean - $task"
    sbatch -p g48 --output=logs/slurm-%j-$task-addition-mean-neox.out run_experiment.sh \
        --experiment_type=addition-mean \
        --model_name=EleutherAI/gpt-neox-20b \
        --task=$task \
        --results_path=../../results/ \
        --list_extraction_layers=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43 \
        --list_beta_averaging=0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0 \
        --mean_activations_path=../../Mean_Activation_Vectors/Generated_Vectors/mean_activations_gpt-neox-20b.pkl
done
