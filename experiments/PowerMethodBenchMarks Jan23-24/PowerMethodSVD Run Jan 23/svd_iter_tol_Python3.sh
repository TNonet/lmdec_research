#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00 
#SBATCH --mem=256G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --job-name=svd_iter_tol_Python3
#SBATCH --output=svd_iter_tol_Python3_out.txt
#SBATCH --error=svd_iter_tol_Python3_err.txt
#SBATCH --constraint="centos7"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tim.nonet@gmail.com

module load python/3.6.3
module load sloan/python/modules/3.6

xvfb-run -d python3 /home/tnonet/lmdec/iter_tol.py