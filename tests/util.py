import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_output_dir(*subdir_names):
    output_dir = os.path.join(CURRENT_DIR, "Outputs", *subdir_names)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir
