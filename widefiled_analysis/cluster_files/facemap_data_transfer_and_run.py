import os
import sys
import yaml
import shutil
import subprocess


def run_facemap(session, vid_path, proc_path, result_folder, python_script):

    command = f" {python_script} {vid_path} {proc_path} {result_folder}"
    print(f"Executing command: {command}")
    subprocess.run(["echo", f"INFO: Launching wf analysis for session {session}"])
    os.system(
        f"sbatch --job-name={session} --export=SESSION={session},SCRIPT={python_script},VID={vid_path},PROC={proc_path},DEST={result_folder} /home/$(whoami)/NWB_analysis/widefiled_analysis/cluster_files/launch_facemap.sbatch")


def transfer_data_and_run_facemap():
    group = sys.argv[1]
    state = sys.argv[2]

    user = 'rdard'
    server_path = f"/home/{user}/servers/"
    facemap_folder = f"/scratch/{user}/facemap"

    facemap_video_folder = os.path.join(facemap_folder, 'videos')
    if not os.path.exists(facemap_video_folder):
        os.makedirs(facemap_video_folder)

    facemap_default_proc = os.path.join(facemap_folder, 'proc_default')
    if not os.path.exists(facemap_default_proc):
        os.makedirs(facemap_default_proc)

    facemap_results = os.path.join(facemap_folder, 'results')
    if not os.path.exists(facemap_results):
        os.makedirs(facemap_results)

    config_file = f"/home/{user}/servers/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_{group}_{state}.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)
    sessions_id = config_dict['Session id']

    for session_id in sessions_id:
        mouse_id = session_id[0:5]
        video_data_path = os.path.join(server_path, "data", mouse_id, "Recording", "Video", session_id,
                                       f'{session_id}_sideview.avi')
        facemap_default_proc_file_path = os.path.join(server_path, "analysis", "Robin_Dard", "proc_default",
                                                      "sideview_proc.npy")

        print(f"Session : {session_id} transfer and run")
        if not os.path.exists(os.path.join(facemap_video_folder, f'{session_id}_sideview.avi')):
            shutil.copy(video_data_path, facemap_video_folder)
        if not os.path.exists(os.path.join(facemap_default_proc, 'sideview_proc.npy')):
            shutil.copy(facemap_default_proc_file_path, facemap_default_proc)

        run_facemap(session_id,
                    os.path.join(facemap_video_folder, f'{session_id}_sideview.avi'),
                    os.path.join(facemap_default_proc, 'sideview_proc.npy'),
                    facemap_results,
                    python_script='facemap_cluster.py')


if __name__ == "__main__":
    transfer_data_and_run_facemap()
