#!/bin/sh
#SBATCH --job-name=net_train  # Job name
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=josebouza@ufl.edu # Where to send mail
#SBATCH --account=bubenik
#SBATCH --qos=bubenik
#SBATCH --nodes=1                   # Use one node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --partition=gpu
#SBATCH --gres=gpu:geforce:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2gb           # Memory per processor
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=array_%A-%a.out    # Standard output and error log
#SBATCH --array=1-2                # Array range
# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users.
pwd; hostname; date

PYTHON=/blue/bubenik/josebouza/Projects/tda-nn/env/bin/python3

#Set the number of runs that each SLURM task should do
PER_TASK=1

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=END_NUM; run++ )); do
    echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run
    mkdir -p ./network_${run}
    PYTHONPATH=./:../../../tda-nn/:../ $PYTHON ../trainer.py \
        --model /blue/bubenik/josebouza/Projects/tda-nn/experiments/1015_graph_geodesic/models/model \
        --csv-file ../data/disk6.csv \
        --training-threshold 1.1 \
        --testing-threshold 0.99
        --batch-size 512 \
        --max-epochs 8000 \
        --learning-rate 0.01 \
        --output-name ./network_$run/network.pt \
        --log-name ./network_$run/training_log.txt \
        ;
  
done

date
