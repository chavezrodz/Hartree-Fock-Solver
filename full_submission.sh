#! /bin/bash

jid1=$(sbatch submission_trials)

sbatch --dependency=afterok:$jid1  submission_final