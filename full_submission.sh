#! /bin/bash

jid1=$(sbatch submission_trials)
#echo $jid1
sbatch --dependency=afterok:$jid1  submission_final


