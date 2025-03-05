import sys
import numpy as np
from facemap import process


def main(vid_path, proc_path, saving_folder):
    proc = np.load(proc_path, allow_pickle=True).item()
    process.run([[vid_path]], proc=proc, savepath=saving_folder)


if __name__ == '__main__':
    vid = sys.argv[1]
    proc = sys.argv[2]
    saving_folder = sys.argv[3]

    main(vid, proc, saving_folder)

