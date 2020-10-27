# Simple PyTorch XOR
Adapted from https://github.com/bdusell/singularity-tutorial

Adaptations include input/output/logging interface recommended
to ease use with UA HPC.

PBS script train_xor.pbs provided as example for running 
train_xor on UA HPC.

The following steps walk through setting up and running the script as a job.
- Log in to the HPC -- the following assumes you are running on elgato
- Clone the code-examples repo in your preferred location within your home directory
- Ensure the <code-examples>/pytorch-xor/xor.txt data is copied to (may already be there, but if not, copy xor.txt to this location (using ‘cp’)):
-- /xdisk/claytonm/projects/code-examples/pytorch-xor
- Execute the <code-examples>/pytorch-xor/train_xor.pbs script (the following assumes you’re in the same directory as the PBS script, ‘$’ denotes the command-line):
-- `$ qsub train_xor.pbs`
-- This will return the job id, where the number is the job id, followed by indication of which HPC resource you are running on (e.g., elgato-adm)
- At this point you can check the job status using qpeek
-- `$ qpeek <job id>`
-- Since this script executes so quickly, it may already be complete, in which case you’ll get a message like:
qstat: Unknown Job Id 207766.elgato-adm
- If the script has executed successfully, navigate to the xdisk results location:
-- `$ cd /xdisk/claytonm/projects/code-examples/pytorch-xor`
- There you’ll see a new directory starting with ‘results_” followed by a timestamp (e.g., results_20201027_172931896698)
Inside, you’ll see the generated log.txt, results.csv and xor_model.pt files.
