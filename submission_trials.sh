#!/bin/bash
#SBATCH -p nodes                          # CPU nodes partition
#SBATCH --job-name=your_job_name ## this helps identify your job in the queue, especially if you have multiple
#SBATCH --nodes=1                       # Use 1 node
#SBATCH --ntasks=1 ## this is the number of nodes you are using. unless you are skilled in using MPI for across-nodes parallelization, this is usually 1

#SBATCH --cpus-per-task=8           # Cores per task
#SBATCH --cpus-per-task=28 ## this is the number of cores you are using on a given node. LISA has 28 cores per node.

#SBATCH --mem-per-cpu=4GB       # Memory (RAM) per core. this is the memory per core. sometimes you need more memory: the total memory per node is 128GB, and mem-per-core * cores < 128 GB for obvious reasons. if you need more memory, you will have to use less cores.
#SBATCH --array=0-8                     # Array Range Initial-Final:Step
#SBATCH --output=Trials_log.txt ## this prints the shell output into a txt for you to consult later

module purge ## this clears the modules loaded before to get a fresh start
module add anaconda3                   # modules to be loaded
python3 main.py $SLURM_ARRAY_TASK_ID 

                                                                                                                                                     