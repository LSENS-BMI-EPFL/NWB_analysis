import sys
from facemap import process

def main(vid_path, proc_path):
    process.run(vid_path, proc=proc_path)


if __name__ == '__main__':
    vid = sys.argv[1]
    proc = sys.argv[2]

    main(vid, proc)