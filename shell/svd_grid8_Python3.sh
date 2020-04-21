#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4
#SBATCH --time=5:00:00
#SBATCH --mem=300G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --job-name=svd_matrix_test_8_Python3
#SBATCH --output=svd_matrix_test_8_Python3_out.txt
#SBATCH --error=svd_matrix_test_8_Python3_err.txt
#SBATCH --constraint="centos7"

module load python/3.6.3
module load sloan/python/modules/3.6

xvfb-run -d python3 /home/tnonet/lmdec/matrix_test_FEB29.py 300 4