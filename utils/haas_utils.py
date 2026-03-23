import os
import sys
import platform
from pathlib import Path

def haas_pathfun(p):
    server_path=[]
    haas_path=[]

    if platform.node() == 'lsens-haas056.intranet.epfl.ch':
        if 'analysis' in p:
            haas_path = Path("/mnt", "lsens-analysis")
            server_path = Path("//sv-nas1.rcp.epfl.ch", "Petersen-Lab", 'analysis')
        elif 'data' in p:
            haas_path = Path("/mnt", "lsens-data")
            server_path = Path("//sv-nas1.rcp.epfl.ch", "Petersen-Lab", 'data')
        else:
            ValueError('Neither "data" nor "analysis" in path')
            return
        
    return Path(p.replace(str(server_path), str(haas_path)))

