#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time 3-0
#SBATCH --mem=0
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=robin.dard@epfl.ch
#SBATCH --export=$SESSION,$SCRIPT,$SOURCE,$DEST
#SBATCH --error=/home/%u/logs/%j.err
#SBATCH --output=/home/%u/logs/%j.out

module load gcc python openmpi
source /home/$USER/venvs/facemap/bin/activate
cd /home/$USER/NWB_analysis/widefiled_analysis/cluster_files

echo "INFO: Running $SCRIPT for session $SESSION with arguments: $VID, $PROC"
python $SCRIPT $VID $PROC