#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00 
#SBATCH --mem=50G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --job-name=svd_matrix_test_6_Python3
#SBATCH --output=svd_matrix_test_6_Python3_out.txt
#SBATCH --error=svd_matrix_test_6_Python3_err.txt
#SBATCH --constraint="centos7"

module load python/3.6.3
module load sloan/python/modules/3.6

xvfb-run -d python3 /home/tnonet/lmdec/matrix_test_FEB28.py 50 8