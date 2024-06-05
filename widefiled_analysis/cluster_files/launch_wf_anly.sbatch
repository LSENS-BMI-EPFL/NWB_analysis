#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time 3-0
#SBATCH --mem 256G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pol.bechvilaseca@epfl.ch
#SBATCH --export=$SESSION,$SCRIPT,$SOURCE,$DEST
#SBATCH --error=/home/%u/logs/%j.err
#SBATCH --output=/home/%u/logs/%j.out

module load gcc python openmpi
source /home/$USER/venvs/nwb_env/bin/activate
cd /home/$USER/NWB_analysis/widefiled_analysis/cluster_files

echo "INFO: Running $SCRIPT for session $SESSION with arguments: $SOURCE, $DEST"
python $SCRIPT $SOURCE $DEST $CLASSIFY $DECODE