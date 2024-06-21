#!/bin/bash
sbatch sso_expts_pnnl_bfgs.sh
sbatch sso_expts_pnnl_fr.sh
sbatch sso_expts_pnnl_sgd.sh

echo "Jobs submitted!"
