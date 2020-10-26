# Example HPC scripts
This directory contains example scripts that have been used to run jobs on the UofA High Performance Compute Cluster. The scripts in this directory show how to submit jobs to be run on these resources, including the commands needed to request compute resources, the loading of necessary modules, execution of singularity images, as well as saving data from both script and program processing. For additional information please see the standard documentation for the UofA clusters available [here](https://public.confluence.arizona.edu/display/UAHPC).

## Script descriptions
In this section we provide brief descriptions on the purpose of each script that can be found in this directory. Where appropriate, we also include a path to the script in question.

#### verify_singularity_img.pbs
This script shows an example of loading and verifying the contents of a singularity image using the ElGato compute cluster. Since ElGato does not have a `debug` queue we use the `windfall` queue for this shorter single-core job. In this job we verify that the singularity image we plan on using later for LaTeX image conversion is executable, contains Python3.8+, contains the Python packages we require for our task, is able to execute `pdflatex`, and can write out result files to `/groups/claytonm`.

## Frequently used HPC docs pages
In this section we provide direct links to some pages in the HPC docs that are frequently used in creating new scripts/jobs on HPC.

#### Resource pages
These pages provide detailed information about the different resources provided by HPC. This ranges from compute, software, and job scheduling tasks.

- [Compute resources](https://public.confluence.arizona.edu/display/UAHPC/Compute+Resources): visit this page to find what resources (CPUs, GPUs, RAM, etc) are available on each of the UofA HPC compute machines. This page also contains plenty of examples of how to request the various specialty resources on each machine.
- [Software resources](https://public.confluence.arizona.edu/display/UAHPC/Software+Resources): visit this page to find a list of all software installed on each of the UofA HPC compute machines. **Super user note --** you can find a more detailed list including the current version number of each software package installed on a machine by logging on to that machine and running the command `module avail`.

#### User guide pages
These pages provide additional guides to help new users get up-to-speed with all of the requirements associated with running HPC resources.

- [Allocation and limits](https://public.confluence.arizona.edu/display/UAHPC/Allocation+and+Limits): visit this page for a great guide on the limits we must respect when using the UofA HPC for storage, compute resources, and job queues.
- [Transferring files](https://public.confluence.arizona.edu/display/UAHPC/Transferring+Files): visit this page for a detailed guide of how to transfer results to/from your local machine and UofA HPC. **Super user note --** the use of `scp`, `sftp`, and `rsync` is the preferred method of transferring data since these options can be built into a script that can be run automatically if required.
- [Running jobs with PBS](https://public.confluence.arizona.edu/pages/viewpage.action?pageId=86409309): visit this page to learn how to create/run job scripts on Ocelote or ElGato -- the older HPC machines that still use the PBS scheduling manager.
- [Running jobs with SLURM](https://public.confluence.arizona.edu/pages/viewpage.action?pageId=93160866): visit this page to learn how to create/run job scripts on Puma -- the newer HPC machien that uses the newer SLURM scheduling manager.
