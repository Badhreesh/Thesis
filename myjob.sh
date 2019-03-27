#!/bin/bash
#PBS -N Jobname
#PBS -l hostlist=^butch,nodes=1:ppn=14:gpus=1:ubuntu1804,mem=5gb,walltime=24:00:00
#PBS -m ae
#PBS -j oe
#PBS -q student
source /misc/student/raob/venv/bin/activate
source /misc/software/cuda/add_environment_cuda9.0.176_cudnnv7.sh
cd /misc/student/raob/baselines/baselines/a2c
python run_doom.py > out.txt

