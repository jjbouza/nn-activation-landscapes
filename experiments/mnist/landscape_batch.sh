#!/bin/sh
#SBATCH --job-name=landscapes  # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=josebouza@ufl.edu # Where to send mail
#SBATCH --account=vemuri
#SBATCH --qos=vemuri-b
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32gb           # Memory per processor
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.out    # Standard output and error log
#SBATCH --array=1-10                # Array range
# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users.
pwd; hostname; date

PYTHON=~/anaconda3/envs/tda-nn/bin/python
export PATH=~/anaconda3/envs/tda-nn/bin/:$PATH
set -e

#Set the number of runs that each SLURM task should do
PER_TASK=1

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

TRAINING_THRESHOLDS="0.7 0.8 0.9 0.95"

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=END_NUM; run++ )); do
    echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
    INPUT_FOLDER=./networks/network_${run}/
    for thresh in $TRAINING_THRESHOLDS; do
        OUTPUT_FOLDER=./landscapes/network_${run}/landscapes_${thresh}/
        mkdir -p $OUTPUT_FOLDER
        mkdir -p $OUTPUT_FOLDER/activations/
        mkdir -p $OUTPUT_FOLDER/activations_class/
        mkdir -p $OUTPUT_FOLDER/activations_plot/
        mkdir -p $OUTPUT_FOLDER/diagrams/
        mkdir -p $OUTPUT_FOLDER/landscapes/
        echo Starting activation with class computations for network ${run} and threshold ${thresh}.
        PYTHONPATH=../../tda-nn/:../ $PYTHON ../../tda-nn/activations.py \
            --network $INPUT_FOLDER/network_${thresh}.pt \
            --input_data ./data/mnist.csv \
            --persistence-class -1 \
            --keep_class \
            --sample-count 100 \
            --layers 0 1 2 3 4 5 6 7\
            --output_dir $OUTPUT_FOLDER/activations_class/ \
            --models_dir /blue/bubenik/josebouza/Projects/tda-nn/experiments/mnist/model/

        echo Starting activation computations for network ${run} and threshold ${thresh}.
        PYTHONPATH=../../tda-nn/:../ $PYTHON ../../tda-nn/activations.py \
            --network $INPUT_FOLDER/network_${thresh}.pt \
            --input_data ./data/mnist.csv \
            --persistence-class -1 \
            --do_softmax \
            --sample-count 100 \
            --layers 0 1 2 3 4 5 6 7\
            --output_dir $OUTPUT_FOLDER/activations/ \
            --models_dir /blue/bubenik/josebouza/Projects/tda-nn/experiments/mnist/model/

        echo Starting activation plotting for network ${run} and threshold ${thresh}.
        PYTHONPATH=../../tda-nn/:../ $PYTHON ../../tda-nn/activations_plot.py \
            --activations $OUTPUT_FOLDER/activations/ \
            --data ./data/mnist.csv \
            --persistence-layers 0 1 2 3 4 5 6 7 \
            --logit-layer 7 \
            --output-dir $OUTPUT_FOLDER/activations_plot/

        echo Starting diagram computations for network ${run} and threshold ${thresh}.
        PYTHONPATH=../../tda-nn:../ $PYTHON ../../tda-nn/diagram.py \
            --activations $OUTPUT_FOLDER/activations/  \
            --max-diagram-dimension 1 1 1 1 1 1 1 1 \
            --diagram-threshold 100000 100000 100000 100000 100000 100000 100000 100000 \
            --persistence-layers 0 1 2 3 4 5 6 7\
            --metric SphereDistance \
            --metric-normalization identity \
            --output-dir $OUTPUT_FOLDER/diagrams/ \

        echo Starting diagram plotting for network ${run} and threshold ${thresh}.
        PYTHONPATH=../../tda-nn:../ $PYTHON ../../tda-nn/diagram_plot.py \
            --diagrams $OUTPUT_FOLDER/diagrams/ \
            --output-dir $OUTPUT_FOLDER/diagram_visualizations/ \

        echo Starting landscape computation for network ${run} and threshold ${thresh}.
        PYTHONPATH=../../tda-nn:../ $PYTHON ../../tda-nn/landscape.py \
            --diagrams $OUTPUT_FOLDER/diagrams/ \
            --landscape-dx 0.005 \
            --landscape-min-x 0 \
            --landscape-max-x 1 \
            --output-dir $OUTPUT_FOLDER/landscapes/ \

        echo Starting landscape plotting for network ${run} and threshold ${thresh}.
        PYTHONPATH=../../tda-nn:../ $PYTHON ../../tda-nn/landscape_plot.py \
            --landscapes $OUTPUT_FOLDER/landscapes/ \
            --output-dir $OUTPUT_FOLDER/landscape_visualizations/ \
            ;
    done
done

date
