import os
import sys
import platform
from pathlib import Path

def haas_pathfun(p):
    if platform.node() == 'haas056.ds-a3-r10.cct.rcp.epfl.ch':
        if 'analysis' in p:
            haas_path = Path("/mnt", "lsens-analysis")
            server_path = Path("//sv-nas1.rcp.epfl.ch", "Petersen-Lab", 'analysis')
        elif 'data' in p:
            haas_path = Path("/mnt", "lsens-data")
            server_path = Path("//sv-nas1.rcp.epfl.ch", "Petersen-Lab", 'analysis')
            
    return Path(p.replace(str(server_path), str(haas_path)))

