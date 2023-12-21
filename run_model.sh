#！/bin/bash
#PBS -q SQUID-H
#PBS --group=G15408
#PBS -l elapstim_req=1:10:10
#PBS -l cpunum_job=76
#PBS -l gpunum_job=8
#PBS -m eb
#PBS -T openmpi
#PBS -M chawitk@hku.hk
#PBS -e $job.err
#PBS -o $job.out
#PBS -r y
cd $PBS_O_WORKDIR
module load BaseGPU/2022
source activate base
nvidia-smi
python ./ShFNO.py >& ./squid/log/ShFNO.txt
## mpirun ${NQSV_MPIOPTS} -np 2 -npernode 2 python $job.py >& $job.txt
