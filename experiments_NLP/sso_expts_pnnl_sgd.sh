#!/bin/bash
#SBATCH --job-name=sgd_sso_expts
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ian.meyer@pnnl.gov
#SBATCH --ntasks=4 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 
#SBATCH --output=/rcfs/projects/sml2024/outs/sgd_outs/sgd%A-%a.log
#SBATCH --error=/rcfs/projects/sml2024/outs/sgd_outs/sgd%A-%a.e
#SBATCH --account=Sml2024
#SBATCH --partition=a100_80_shared
#SBATCH --array=0-15

pwd; hostname; date
echo $(nproc)
module load cuda/11.1
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
echo Sourced conda
conda activate new_soo
echo Activated soo  ##SBATCH --array=0-15
which python3

nvidia-smi -L

# #HOTFIX:
# SLURM_ARRAY_TASK_ID=0
# SLURM_ARRAY_TASK_MAX=16

echo This is task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX 
python sso_expts_pnnl.py --task_id $SLURM_ARRAY_TASK_ID \
    --total_tasks $((SLURM_ARRAY_TASK_MAX + 1)) \
    --optimizer sgd \
    --outdir=./expt_rslts_sgd_linear_probe_no_qlora/ \
    # --force