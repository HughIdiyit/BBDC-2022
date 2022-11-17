import os
import glob
from itertools import groupby
from operator import itemgetter
import argparse

import pandas as pd
import numpy as np
import imageio as iio
from tqdm import tqdm


def create_mocap_dataset(offset=90, max_len=120):
    # List of complete mocap files
    files = [
        "data/s055t03_mocap.csv",
        "data/s056t05_mocap.csv",
        "data/s057t05_mocap.csv",
        "data/s057t06_mocap.csv",
        "data/s058t03_mocap.csv",
        "data/s059t04_mocap.csv",
        "data/s063t02_mocap.csv",
        "data/s064t03_mocap.csv",
        "data/s070t04_mocap.csv"
    ]
    info_csvs = sorted(glob.glob('data/*info.csv'))
    mocap_csvs = sorted(glob.glob('data/*mocap.csv'))
    num_iterations = len(info_csvs)

    # Create test/prediction samples
    for inf, f in tqdm(zip(info_csvs, mocap_csvs), desc="Create Mocap test", total=num_iterations):
        if f not in files:
            current_info = pd.read_csv(inf).iloc[:, 0].to_numpy()
            indices = np.argwhere(np.logical_not(current_info))
            indices = indices.flatten()
            index_tuples = []

            # Find position of False-blocks and their lengths
            for k, g in groupby(enumerate(indices), lambda ix : ix[0] - ix[1]):
                index_tuples.append([_ for _ in map(itemgetter(1), g)])
            index_tuples = [(tp[0], len(tp)) for tp in index_tuples]

            mocap = pd.read_csv(f).to_numpy()
            mocap_samples = []
            # Cut out preceding samples (of history length) for initial prediction
            for tp in index_tuples:
                mocap_samples.append(mocap[tp[0]-offset:tp[0]])
            np.save(f[:-4]+"_samples", mocap_samples)
            np.save(f[:-4]+"_ind_tuples", index_tuples)

    # Create train samples
    mocap_output = []
    for inf, f in tqdm(zip(info_csvs, mocap_csvs), desc="Create Mocap train", total=num_iterations):
        current_info = pd.read_csv(inf).iloc[:, 0].to_numpy()
        indices = np.argwhere(np.equal(current_info, True))
        indices = indices.flatten()
        index_tuples = []

        # Find position of True-blocks and their lengths
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            index_tuples.append([_ for _ in map(itemgetter(1), g)])
        index_tuples = [(tp[0], tp[-1]) for tp in index_tuples]
        mocap = pd.read_csv(f).to_numpy()
        mocap_samples = []
        for tp in index_tuples:
            for i in range(tp[0], tp[1]-max_len, offset):
                mocap_sample = mocap[i:i+max_len]
                if -999999000 in mocap_sample:  # Skip erroneous data
                    continue
                # Normalize and save
                mc_mean = np.mean(mocap_sample)
                mc_std = np.std(mocap_sample)
                mocap_sample = (mocap_sample - mc_mean) / mc_std
                mocap_samples.extend(mocap_sample)
        mocap_output.extend(mocap_samples)

    np.save("all_mocap_samples", mocap_output)


def load_video(path):
    # Open a ffmpeg reader with dimensions 160x96 (video size) of the given path
    reader = iio.get_reader(path, 'ffmpeg', size=(160, 96))
    # Only keep the one channel (they are all the same) and convert to uint8 (ie, values in range 0-255).
    return np.array([frame[:, :, 0] for frame in reader]).astype(np.uint8)


def create_video_dataset(offset, history_len, reverse_history_len):
    info_csvs = sorted(glob.glob('data/*info.csv'))
    videos = sorted(glob.glob("data/*.mp4"))
    num_iterations = len(info_csvs)
    video_output = []

    # Create test/prediction samples
    for inf, vid in tqdm(zip(info_csvs, videos), desc="Create Video test", total=num_iterations):
        vid_array = load_video(vid)
        current_info = pd.read_csv(inf).iloc[:, 1].to_numpy()
        indices = np.argwhere(np.logical_not(current_info))
        indices = indices.flatten()
        index_tuples = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            index_tuples.append([_ for _ in map(itemgetter(1), g)])
        index_tuples = [(tp[0], len(tp)) for tp in index_tuples]

        video_samples = []
        video_reverse = []
        for tp in index_tuples:
            video_samples.append(vid_array[tp[0]-history_len:tp[0]])
            # Cut out following samples (of history length) for initial prediction
            # Wrapped in try-except block because they may not exist
            try:
                samples_behind_false_block = vid_array[tp[0]+tp[1]:tp[0]+tp[1]+reverse_history_len]
                reverse = np.flip(samples_behind_false_block, axis=0)
                video_reverse.append(reverse)
            except:
                pass

        np.save(vid[:-4]+"_reverse", video_reverse)
        np.save(vid[:-4]+"_samples", video_samples)
        np.save(vid[:-4]+"_ind_tuples", index_tuples)

    # Create train samples
    for inf, vid in tqdm(zip(info_csvs, videos), desc="Create Video train", total=num_iterations):
        vid_array = load_video(vid)

        current_info = pd.read_csv(inf).iloc[:, 1].to_numpy()
        indices = np.argwhere(np.equal(current_info, True))
        indices = indices.flatten()
        index_tuples = []
        for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
            index_tuples.append([_ for _ in map(itemgetter(1), g)])
        index_tuples = [(tp[0], tp[-1]) for tp in index_tuples]

        vid_samples = []
        for tp in index_tuples:
            if tp[1]-tp[0] < history_len+offset:
                continue
            for i in range(tp[0], tp[1]-(history_len+offset), history_len+offset):
                vid_sample = vid_array[i:i+history_len+offset]
                vid_samples.append(vid_sample)

        np.save(vid[:-4]+"_train_samples", vid_samples)
        video_output.extend(vid_samples)
    np.save("all_vid_samples", video_output)


def torch_train_test_split(seq_len):
    x = np.load("all_vid_samples.npy", allow_pickle=True)
    x = x.reshape((-1, seq_len, 1, 96, 160))

    train = x[:int(0.9*x.shape[0])]
    val = x[int(0.9*x.shape[0]):]

    np.save("video_train_torch", train)
    np.save("video_val_torch", val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=int)
    parser.add_argument("prediction", type=int)
    parser.add_argument("reverse_history", type=int)
    args = parser.parse_args()

    create_mocap_dataset(offset=105, max_len=106)
    create_video_dataset(offset=args.prediction, history_len=args.history, 
                        reverse_history_len=args.reverse_history)
    total_size = args.prediction + args.history
    torch_train_test_split(total_size)
