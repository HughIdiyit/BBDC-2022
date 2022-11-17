import os

PYTHON_VERSION = "python"  # Enter name of python binary e.g. python3

if __name__ == "__main__":
    print("Preprocessing...")
    os.system(f"{PYTHON_VERSION} preprocessing.py 15 5 15")
    print("Mocap prediction...")
    os.system(f"{PYTHON_VERSION} mocap_prediction.py")
    os.system(f"{PYTHON_VERSION} mocap_postprocessing.py")
    print("Video prediction...")
    os.system(f"{PYTHON_VERSION} vid_pred_twosided.py")
    print("Done")
