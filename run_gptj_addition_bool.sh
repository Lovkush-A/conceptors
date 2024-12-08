# compound_tasks=("english-french&antonyms" "english-french&capitalize" "singular-plural&capitalize")
# compound_tasks=("english-french&antonyms")
compound_tasks=("english-french&capitalize" "singular-plural&capitalize")
# compound_tasks=("singular-plural&capitalize")

for task in "${compound_tasks[@]}"
do
    echo "GPT-J 6B - addition-bool - $task"
    sbatch -p g24 --output=logs/slurm-%j-$task-addition-bool-gptj.out run_merged_experiment.sh \
        --experiment_type=addition \
        --model_name=EleutherAI/gpt-j-6b \
        --task=$task \
        --results_path=../../results/boolean \
        --list_extraction_layers=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
        --list_beta_averaging=0.5,1.0,2.0,3.0,4.0,5.0
        # --list_beta_conceptor=0.5,1.0,2.0,3.0,4.0,5.0 \
        # --list_apertures_normal=0.001,0.0125,0.05,0.1
        # --list_extraction_layers=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
        # --list_beta_conceptor=0.5,1.0,2.0,3.0,4.0,5.0 \
        # --list_apertures_normal=0.001,0.0125,0.05,0.1
done
