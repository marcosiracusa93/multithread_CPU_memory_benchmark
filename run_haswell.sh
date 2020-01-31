#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell

#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=msiracusa@lbl.gov

max_num_threads=$((2**5))
#max_num_trans=$((2**10))
#max_access_length=$((2**18))
max_num_trans=$((2**0))
max_access_length=$((2**28))

export KMP_AFFINITY=compact,verbose

export OMP_PROC_BIND=true
export OMP_PLACES=threads #(core/threads)

srun ./bin/benchmark_mt $max_num_threads $max_num_trans $max_access_length 0

export OMP_NUM_THREADS=$max_num_threads


#for i in {0..9}
#do
#	pow=$((10**$i))
#	srun ./bin/benchmark_mt $threads $pow
#done
