#!/bin/bash

# Submit this script with: sbatch test.sh

#SBATCH --time=20:00:00   # how much time you want in hours:minutes:seconds
#SBATCH --ntasks=28   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "Batch_3_1000"   # whatever job name
#SBATCH --mail-user=mkapasia@caltech.edu   # email address to receive email on process end
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE (activate your environment and run the file)
# source /home/mkapasia/Y/envs/activate modeling_3.12
/home/mkapasia/anaconda3/envs/modeling_3.12/bin/python /resnick/groups/murray-biocircuits/mkapasia/Finetuning/Python_Scripts/Fine_tuning_on_each_experimental_condition_separately_batch_3_fewer_steps/Fine_tuning_on_each_experimental_condition_separately_batch_3_1000_steps.py
