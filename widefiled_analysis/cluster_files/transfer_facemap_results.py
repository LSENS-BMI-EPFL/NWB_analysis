import os
import sys
import json
import shutil


def transfer_results():

    result_folder = f"/scratch/rdard/facemap/results"
    dest_folder = f"/home/rdard/servers"
    vid_folder = f"/scratch/rdard/facemap/videos"

    for result in os.listdir(result_folder):

        session_id = "_".join(result.split('_')[0:3])

        facemap_path = os.path.join(result_folder, result)
        vid_path = os.path.join(vid_folder, f'{session_id}_sideview.avi')

        # if os.path.exists(vid_path):
        #     os.remove(vid_path)

        output_folder = os.path.join(dest_folder, "analysis", "Robin_Dard", 'facemap_cluster_results')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        print(f'Session: {session_id}, result : {result}')
        print(f"Copying data from: {facemap_path} to: {output_folder}")

        shutil.copyfile(facemap_path, os.path.join(output_folder, result))

    return


if __name__ == "__main__":
    transfer_results()
