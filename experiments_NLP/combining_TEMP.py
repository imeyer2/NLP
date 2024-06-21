import os
import shutil
from pathlib import Path

"""Combine all the various locations where JSON files are..... put into a final directory"""

def get_creation_time(file_path):
    return os.path.getctime(file_path)

def combine_directories(dir_a, dir_b, dir_c):
    Path(dir_c).mkdir(parents=True, exist_ok=True)

    files_a = {f: os.path.join(dir_a, f) for f in os.listdir(dir_a) if f.endswith('.json')}
    files_b = {f: os.path.join(dir_b, f) for f in os.listdir(dir_b) if f.endswith('.json')}

    all_files = set(files_a.keys()).union(set(files_b.keys()))

    for file in all_files:
        file_a_path = files_a.get(file)
        file_b_path = files_b.get(file)

        if file_a_path and file_b_path:
            if get_creation_time(file_a_path) > get_creation_time(file_b_path):
                shutil.copy2(file_a_path, os.path.join(dir_c, file))
            else:
                shutil.copy2(file_b_path, os.path.join(dir_c, file))

        elif file_a_path:
            shutil.copy2(file_a_path, os.path.join(dir_c, file))
        
        elif file_b_path:
            shutil.copy2(file_b_path, os.path.join(dir_c, file))



# combine_directories("/rcfs/projects/sml2024/pytorch_soo/experiments/expt_rslts_bfgs_linear_probe_no_qlora",
#                     "/rcfs/projects/sml2024/pytorch_soo/experiments/expt_rslts",
#                     dir_c = "/rcfs/projects/sml2024/pytorch_soo/experiments/COMBINED")

combine_directories("/rcfs/projects/sml2024/pytorch_soo/experiments/expt_rslts",
                    "/rcfs/projects/sml2024/pytorch_soo/experiments/expt_rslts_fr_no_qlora_linear_probe",
                    dir_c = "/rcfs/projects/sml2024/pytorch_soo/experiments/COMBINED")

combine_directories("/rcfs/projects/sml2024/pytorch_soo/experiments/COMBINED",
                    "/rcfs/projects/sml2024/pytorch_soo/experiments/expt_rslts_sgd_linear_probe_no_qlora",
                    dir_c = "/rcfs/projects/sml2024/pytorch_soo/experiments/all_jsons")

