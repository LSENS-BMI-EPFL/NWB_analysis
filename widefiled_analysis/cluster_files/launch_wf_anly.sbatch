#!/bin/bash -l
#SBATCH --export=SESSION,$SCRIPT,$SOURCE,$DEST
#SBATCH --job-name=${JOBNAME}
#SBATCH --nodes=1
#SBATCH --time 3-0
#SBATCH --mail-type=END,FAIL
#SBATCH --error=/home/%u/logs/${SESSION}.err
#SBATCH --output=/home/%u/logs/${SESSION}.out
#SBATCH --mail-user=pol.bechvilaseca@epfl.ch

module load gcc python openmpi
source /home/${USER}/venvs/nwb_env/bin/activate
cd /home/${USER}/NWB_analysis/widefield_analysis/cluster_files

echo "INFO: Running ${SCRIPT} for session ${SESSION} with arguments: ${SOURCE}, ${DEST}"
python ${SCRIPT} ${SOURCE} ${DEST}