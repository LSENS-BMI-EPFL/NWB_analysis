import os
import sys
import yaml
import shutil
import subprocess


def transfer_data():
    user = sys.argv[1]
    group = sys.argv[2]
    python_script = sys.argv[3]

    config_file = f"/home/{user}/servers/analysis/{'Pol_Bech' if user == 'bechvila' else 'Robin_Dard'}/group.yaml"
    with open(config_file, 'r', encoding='utf8') as stream:
        config_dict = yaml.safe_load(stream)

    local_path = "//sv-nas1.rcp.epfl.ch/Petersen-Lab/"
    server_path = f"/home/{user}/servers/"
    NWB_folder = f"/scratch/{user}/NWB"
    result_folder = f"/scratch/{user}/wf_results"

    for session, nwb_path in config_dict['NWB_CI_LSENS'][group]:

        nwb_path = nwb_path.replace(local_path, server_path).replace("\\", "/")
        print(f"Transferring {session} to {NWB_folder}")
        if not os.path.exists(os.path.join(NWB_folder, session+'.nwb')):
            shutil.copy(nwb_path, NWB_folder)

        for decode in ['baseline', 'stim']:
            for classify_by in ['context', 'lick', 'tone']:
                dest_folder = os.path.join(result_folder, python_script.split(".")[0], decode, session.split("_")[0], session).replace("\\", "/")
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder, exist_ok=True)

                session_to_anly = os.path.join(NWB_folder, session + '.nwb').replace('\\', '/')
                command = f" {python_script} {session_to_anly} {dest_folder}"
                print(f"Executing command: {command}")
                subprocess.run(["echo", f"INFO: Launching wf analysis for session {session}"])
                os.system(f"sbatch --job-name={session} --export=SESSION={session},SCRIPT={python_script},SOURCE={session_to_anly},DEST={dest_folder},CLASSIFY={classify_by},DECODE={decode} /home/bechvila/NWB_analysis/widefiled_analysis/cluster_files/launch_wf_anly.sbatch")

if __name__ == "__main__":
    transfer_data()
