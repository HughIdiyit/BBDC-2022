import math
import glob

import numpy as np
import torch
import imageio as iio
from tqdm import tqdm

from video_model import ConvLSTM, VideoModel

offset = 15  # history
stride = 5  # pred
reverse_offset = 15
reverse_stride = 5
total_size = stride + offset
reverse_total_size = reverse_stride + reverse_offset

# input_chan, hidden_chan, kernel_size, batch_first, bias, return_all_layers
forward_convlstm = ConvLSTM(1, [16, 18, 20], (5, 5), 3, True, True, False)
backward_convlstm = ConvLSTM(1, [16, 18, 20], (5, 5), 3, True, True, False)
model_forward = VideoModel(forward_convlstm, (3, 3), offset, stride)
model_forward.load_state_dict(torch.load("best_model_reverse/forward.pt"))
model_backward = VideoModel(backward_convlstm, (3, 3), reverse_offset, reverse_stride)
model_backward.load_state_dict(torch.load("best_model_reverse/backward.pt"))

device = torch.device("cuda")
model_forward.to(device)
model_backward.to(device)
model_forward.eval()
model_backward.eval()


def predict_block(block_len, prediction, model, _offset, _stride, _total_size):
    with torch.no_grad():
        for idx in range(0, block_len, _stride):
            test_data = prediction[idx:(idx + _offset)]
            test_data = test_data.reshape((1, _offset, 1, 96, 160))
            test_data = test_data.astype(np.float32)
            test_data = torch.tensor(test_data).to(device)

            model_pred = model(test_data)
            model_pred = torch.reshape(model_pred, (_stride, 96, 160))
            model_pred = model_pred.cpu().numpy()
            if prediction[(idx + _offset):].shape[0] < _stride:
                pred_shape = prediction[(idx + _offset):].shape[0]
                prediction[idx + _offset:] = model_pred[:pred_shape]
            else:
                prediction[idx + _offset:(idx + _total_size)] = model_pred

    prediction = prediction * 255
    prediction = prediction[_offset:]
    prediction = prediction.astype(np.uint8)

    return prediction


samples = sorted(glob.glob("data/*video_samples.npy")) 
reverse = sorted(glob.glob("data/*video_reverse.npy"))  
ind_tuples = sorted(glob.glob("data/*video_ind_tuples.npy"))

global_predictions = []

for (s, r, i) in tqdm(zip(samples, reverse, ind_tuples), desc="Blocks", total=len(samples)):
    data = np.load(s)
    data_reverse = np.load(r)
    tuples = np.load(i)

    for (block, block_reverse, tp) in zip(data, data_reverse, tuples):

        block = block / 255
        block_reverse = block_reverse / 255

        block_len = tp[1]

        half_length = block_len / 2
        forward_len = math.ceil(half_length)
        backward_len = math.floor(half_length)

        forward_pred = np.zeros((forward_len+offset, 96, 160))
        backward_pred = np.zeros((backward_len+reverse_offset, 96, 160))

        forward_pred[:offset] = block
        backward_pred[:reverse_offset] = block_reverse

        forward_pred = predict_block(forward_len, forward_pred, model_forward, offset, stride, total_size)
        backward_pred = predict_block(backward_len, backward_pred, model_backward, reverse_offset, reverse_stride,
                                      reverse_total_size)
        backward_pred = np.flip(backward_pred, axis=0)

        prediction = np.concatenate((forward_pred, backward_pred))

        global_predictions.extend(prediction)

np.save('video_predictions.npy', global_predictions)


def write_video(path, arr):
    # Open an ffmpeg writer
    writer = iio.get_writer(path, fps=30, format='ffmpeg')

    for im in arr:
        # Add each frame from the passed array
        writer.append_data(im.astype(np.uint8))

    # Close the file again
    writer.close()


write_video(f'video.mp4', global_predictions)
