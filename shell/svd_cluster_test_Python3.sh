#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --mem=220G
#SBATCH --partition=sched_mit_sloan_interactive
#SBATCH --job-name=svd_cluster_test_Python3
#SBATCH --output=svd_cluster_test_Python3_out.txt
#SBATCH --error=svd_cluster_test_Python3_err.txt
#SBATCH --constraint="centos7"

module load python/3.6.3
module load sloan/python/modules/3.6

xvfb-run -d python3 /home/tnonet/lmdec/matrix_test_MARCH1.py