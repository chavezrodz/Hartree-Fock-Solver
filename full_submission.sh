#! /bin/bash
RES=$(sbatch submission_trials) && sbatch --dependency=afterok:${RES##* } submission_final
