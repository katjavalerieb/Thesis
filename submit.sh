#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J dl_.BoneCon0.1
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 GPU in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- request system memory --
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
#BSUB -u kvabo@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

# Print GPU status
nvidia-smi

# Load the CUDA module
module load cuda/11.6

# Activate your Python environment
source myenvv3.11.7/bin/activate

# Run device query to check GPU availability and capabilities
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

# Run your Python training script with arguments
time python qcDataTargetContinued.py --targetDSC 1.0