#!/bin/bash
#SBATCH --job-name=bfgs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ian.meyer@pnnl.gov
#SBATCH --ntasks=4 
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00 
#SBATCH --output=/rcfs/projects/sml2024/outs/bfgs_outs/bfgs%A-%a.log
#SBATCH --error=/rcfs/projects/sml2024/outs/bfgs_outs/bfgs%A_%a.e
#SBATCH --account=Sml2024
#SBATCH --partition=a100_shared
#SBATCH --array=0-15

pwd; hostname; date
echo $(nproc)
# module load cuda/11.1
module load python/miniconda3.9
source /share/apps/python/miniconda3.9/etc/profile.d/conda.sh
echo Sourced conda
conda activate new_soo
echo Sourced environment
echo Activated new_soo  ##SBATCH --array=0-15

# export PATH=$PATH:/usr/local/cuda-11.4/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/bin 
# export FORCE_CUDA="1" #FORCE_CUDA will make pytorch act like a GPU is available always, even on a CPU.

which python3
nvidia-smi

# #HOTFIX:
# SLURM_ARRAY_TASK_ID=0
# SLURM_ARRAY_TASK_MAX=16
echo This is task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_MAX
python3 sso_expts_pnnl.py --task_id $SLURM_ARRAY_TASK_ID \
        --total_tasks $((SLURM_ARRAY_TASK_MAX + 1)) \
        --optimizer bfgs \
        --outdir ./expt_rslts_bfgs_linear_probe_no_qlora/ \

