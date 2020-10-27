#!/bin/bash

# --------------------------------------------------------------
### PART 1: Requests resources to run your job.
# --------------------------------------------------------------
### Set the job name (needed for job lookup)
#PBS -N pytorch_xor
### Specify the PI group for this job (always claytonm)
#PBS -W group_list=claytonm
### Set the queue for your job (options: windfall, standard, high_priority, debug -- only on Ocelote)
#PBS -q windfall
### Set the number of nodes, cores and memory that will be used for this job
### pcmem is optional as it defaults to 6gb per core. Note: mem=ncpus x pcmem
#PBS -l select=1:ncpus=1:mem=4gb:pcmem=4gb
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=0:3:0
### Optional. cput = time x ncpus. If not included, default := cput = walltime x ncpus.
#PBS -l cput=0:3:0

# --------------------------------------------------------------
### PART 2: Executes bash commands to run your job
# --------------------------------------------------------------
### Load required modules/libraries if needed
module load singularity/3.6.4

### Define path to singularity image that we will use for running commands
CONTAINER=/groups/claytonm/containers/torch-base_latest.sif

### NOTE: You must update the SCRIPT_PATH to your context
SCRIPT_ROOT=/home/u25/claytonm/projects/code-examples/pytorch-xor
DATA_ROOT=/xdisk/claytonm/projects/code-examples/pytorch-xor/xor.txt
OUTPUT_ROOT=/xdisk/claytonm/projects/code-examples/pytorch-xor/results

### Run commands in the singularity container using exec.
cd $SCRIPT_ROOT
singularity exec $CONTAINER python3 train_xor.py --data-file=$DATA_ROOT --output-root=$OUTPUT_ROOT --gen-output-timestamp --save-model
