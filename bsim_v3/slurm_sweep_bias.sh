#!/bin/bash
#SBATCH --job-name=bias-sweep-2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

#SBATCH --array=1-1000

#SBATCH --output=logs/slurm-%A_%a_%N_%x.out 	  # Filename pattern: https://slurm.schedmd.com/sbatch.html#lbAH
#SBATCH --time=0-03:00:00


# Additional (smaller) partitions can be scheduled as follows: `sbatch --array=1-300 --cpus-per-task=3 --partition=nocard,smallcard slurm_job.sh`

##SBATCH --partition=midcard,dualcard,bigcard
##SBATCH --nodes=1                # node count - unles you are VERY good at what you're doing, you should keep this as-is
##SBATCH --ntasks=1               # total number of tasks across all nodes - you only have 1 node, so you only have 1 task. Leave this.
###SBATCH --gres=gpu:1 #most machines have a single GPU, so leave this as-is. If you are on a dual GPU partition, this can be changed to --gres=gpu:2 to use both
###SBATCH --mem-per-cpu=12G         # memory per cpu-core - unless you're doing something obscene the default here is fine. This is RAM, not VRAM, so it's like storage for your dataset
##SBATCH --exclusive # reserve the entire node

# Environment variable referece:
# https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES

#source /datasets_nfs/watonomous/ben/init-shell.sh

set -o pipefail -o errexit -o nounset

__expdir=exp/bsim_v3/${SLURM_JOB_NAME}
mkdir -p "$__expdir"

# conda activate doesn't work well in strict mode.jk
# https://github.com/conda/conda/issues/8186#issuecomment-532874667
set +o nounset +o errexit
if [ -f $HOME/miniconda3/etc/profile.d/conda.sh ]; then source $HOME/miniconda3/etc/profile.d/conda.sh; fi
conda activate research
set -o nounset -o errexit

python bsim_v3/run_sim.py run-multiple --fault-type bias --out-file "$__expdir"/results-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID:-0}-$(hostname).parquet
