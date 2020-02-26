#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=5
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --constraint=haswell

#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=marco.siracusa@mail.polimi.it

max_num_elements=$((2**28))
num_threads=$((2**5))

export KMP_AFFINITY=compact,verbose

export OMP_PROC_BIND=true
export OMP_PLACES=threads #(core/threads)
export OMP_NUM_THREADS=$num_threads

srun ./bin/benchmark_mt $num_threads $max_num_elements 0


#for i in {0..9}
#do
#	pow=$((10**$i))
#	srun ./bin/benchmark_mt $threads $pow
#done
