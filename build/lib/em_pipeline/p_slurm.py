def get_slurm(cmd,):
    pref ="""#!/bin/bash
    #SBATCH -N 1 # number of nodes
    #SBATCH -p cox
    #SBATCH -n 1 # number of cores
    #SBATCH --mem 10000 # memory pool for all cores
    #SBATCH -t 1-00:00 # time (D-HH:MM)
    #SBATCH -o """+Do+"""slurm.%N.%j.out # STDOUT
    #SBATCH -e """+Do+"""slurm.%N.%j.err # STDERR
    """
