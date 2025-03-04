import os
import sys
import yaml
import shutil
import subprocess


def launch_decoding(state, group, session, NWB_folder, python_script, result_folder):
    for decode in ['baseline']:
        for classify_by in ['context']:
            dest_folder = os.path.join(result_folder, python_script.split(".")[0] + f"_{group}_{state}", decode,
                                       session.split("_")[0], session).replace("\\", "/")
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder, exist_ok=True)

            session_to_anly = os.path.join(NWB_folder, session + '.nwb').replace('\\', '/')
            command = f" {python_script} {session_to_anly} {dest_folder}"
            print(f"Executing command: {command}")
            subprocess.run(["echo", f"INFO: Launching wf analysis for session {session}"])
            os.system(
                f"sbatch --job-name={session} --export=SESSION={session},SCRIPT={python_script},SOURCE={session_to_anly},DEST={dest_folder},CLASSIFY={classify_by},DECODE={decode} /home/bechvila/NWB_analysis/widefiled_analysis/cluster_files/launch_decoding.sbatch")


def launch_decoding_parallel(state, group, session, NWB_folder, python_script, result_folder):
    for decode in ['baseline']:
        for classify_by in ['context']:
            dest_folder = os.path.join(result_folder, python_script.split(".")[0] + f"_{group}_{state}", decode,
                                       session.split("_")[0], session).replace("\\", "/")
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder, exist_ok=True)

            session_to_anly = os.path.join(NWB_folder, session + '.nwb').replace('\\', '/')
            command = f" {python_script} {session_to_anly} {dest_folder}"
            print(f"Executing command: {command}")
            subprocess.run(["echo", f"INFO: Launching wf analysis for session {session}"])
            os.system(
                f"sbatch --job-name={session} --export=SESSION={session},SCRIPT={python_script},SOURCE={session_to_anly},DEST={dest_folder},CLASSIFY={classify_by},DECODE={decode} /home/bechvila/NWB_analysis/widefiled_analysis/cluster_files/launch_decoding_parallel.sbatch")


def launch_correlations(state, group, session, NWB_folder, python_script, result_folder):

    dest_folder = os.path.join(result_folder, python_script.split(".")[0] + f"_{group}_{state}",
                               session.split("_")[0], session).replace("\\", "/")

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder, exist_ok=True)

    session_to_anly = os.path.join(NWB_folder, session + '.nwb').replace('\\', '/')

    command = f" {python_script} {session_to_anly} {dest_folder}"
    print(f"Executing command: {command}")
    subprocess.run(["echo", f"INFO: Launching wf analysis for session {session}"])
    os.system(
        f"sbatch --job-name={session} --export=SESSION={session},SCRIPT={python_script},SOURCE={session_to_anly},DEST={dest_folder} /home/bechvila/NWB_analysis/widefiled_analysis/cluster_files/launch_correlations.sbatch")


def run_facemap(session, vid_path, proc_path, python_script):

    command = f" {python_script} {vid_path} {proc_path}"
    print(f"Executing command: {command}")
    subprocess.run(["echo", f"INFO: Launching wf analysis for session {session}"])
    os.system(
        f"sbatch --job-name={session} --export=SESSION={session},SCRIPT={python_script},VID={vid_path},PROC={proc_path} /home/$(whoami)/NWB_analysis/widefiled_analysis/cluster_files/launch_facemap.sbatch")


def transfer_data():
    user = sys.argv[1]
    group = sys.argv[2]
    python_script = sys.argv[3]

    for state in ['naive', 'expert']:
        config_file = f"/home/{user}/servers/z_LSENS/Share/Pol_Bech/Session_list/context_sessions_{group}_{state}.yaml"
        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        local_path = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/"
        server_path = f"/home/{user}/servers/"
        NWB_folder = f"/scratch/{user}/NWB"
        result_folder = f"/scratch/{user}/wf_results"

        for i, nwb_path in enumerate(config_dict['Session path']):
            session = config_dict['Session id'][i]

            nwb_path = nwb_path.replace(local_path.replace("/", "\\"), server_path).replace("\\", "/")
            print(f"Transferring {session} to {NWB_folder}")
            if not os.path.exists(os.path.join(NWB_folder, session+'.nwb')):
                shutil.copy(nwb_path, NWB_folder)

            if python_script == 'widefield_decoding_cluster.py':
                launch_decoding(state, group, session, NWB_folder, python_script, result_folder)
            elif python_script == 'widefield_decoding_cluster_parallel.py' or python_script == 'widefield_decoding_cluster_parallel_synthetic.py':
                launch_decoding_parallel(state, group, session, NWB_folder, python_script, result_folder)
            elif python_script == 'pixel_cross_correlation.py':
                launch_correlations(state, group, session, NWB_folder, python_script, result_folder)
            else:
                subprocess.run(["echo", f"INFO: No session was launched, wrong script"])


def transfer_data_and_run_facemap():
    # user = sys.argv[1]
    user = 'rdard'
    # group = sys.argv[2]
    # python_script = sys.argv[3]

    server_path = f"/home/{user}/servers/"
    local_path = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/"
    facemap_folder = f"/scratch/{user}/facemap"

    files = os.listdir(os.path.join(server_path, "analysis", "Robin_Dard", "facemap_gfp_experts"))
    files = [file for file in files if 'npy' in file]

    for file in files:
        mouse_id = file[0:5]
        session_id = file[0:21]
        video_data_path = os.path.join(server_path, "data", mouse_id, "Recording", "Video", session_id,
                                       f'{session_id}_sideview.avi')
        facemap_proc_file_path = os.path.join(server_path, "analysis", "Robin_Dard", "facemap_gfp_experts",
                                              f'{session_id}_sideview_proc.npy')

        print(f"Session : {session_id} transfer and run")
        if not os.path.exists(os.path.join(facemap_folder, f'{session_id}_sideview.avi')):
            shutil.copy(video_data_path, facemap_folder)
        if not os.path.exists(os.path.join(facemap_folder, f'{session_id}_sideview_proc.npy')):
            shutil.copy(facemap_proc_file_path, facemap_folder)

        run_facemap(session_id, video_data_path, facemap_proc_file_path, python_script='facemap_cluster.py')


if __name__ == "__main__":
    # transfer_data()
    transfer_data_and_run_facemap()