#!/bin/bash      ## this line tells the script what to load at start (you can edit your bash file to automatically load certain modules, and so on)

#SBATCH --job-name=your_job_name ## this helps identify your job in the queue, especially if you have multiple

#SBATCH --output=your_output_text_file.txt ## this prints the shell output into a txt for you to consult later

#SBATCH --ntasks=1 ## this is the number of nodes you are using. unless you are skilled in using MPI for across-nodes parallelization, this is usually 1

#SBATCH --cpus-per-task=28 ## this is the number of cores you are using on a given node. LISA has 28 cores per node.
                                                                                                                                                     
#SBATCH --mem-per-core=4GB ## this is the memory per core. sometimes you need more memory: the total memory per node is 128GB, and mem-per-core * cores < 128 GB for obvious reasons. if you need more memory, you will have to use less cores.

module purge ## this clears the modules loaded before to get a fresh start
module add anaconda3 ## this adds your preferred computation modules, in my case Python 2.7
python3 testing_script.py ## this executes your actual Python script
