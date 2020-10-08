#! /bin/bash
RES=$(qsub submission_trials_PBS) && qsub -W depend=afterok:${RES##* } submission_final_PBS
