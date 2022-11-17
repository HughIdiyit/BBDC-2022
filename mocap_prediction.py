import glob

import numpy as np
import tensorflow as tf

HEADER = "HeadTop_X,HeadTop_Y,HeadTop_Z,HeadFront_X,HeadFront_Y,HeadFront_Z,HeadSide_X,HeadSide_Y,HeadSide_Z"

model = tf.keras.models.load_model('best_model_mocap')

samples = sorted(glob.glob("data/*mocap_samples.npy"))
ind_tuples = sorted(glob.glob("data/*mocap_ind_tuples.npy"))

global_predictions = []
offset = 105
seq_length = 1
total_size = offset + seq_length

for (s, i) in zip(samples, ind_tuples):
    data = np.load(s)
    tuples = np.load(i)

    for (block, tp) in zip(data, tuples):
        mc_mean = np.mean(block)
        mc_std = np.std(block)
        block_norm = (block - mc_mean) / mc_std

        block_len = tp[1]

        prediction = np.zeros((block_len + offset, 9))
        prediction[:offset] = block_norm

        for idx in range(0, block_len, seq_length):
            test_data = prediction[idx:(idx + offset)]
            test_data = test_data.reshape((1, offset, 9))

            model_pred = model(test_data)
            model_pred = tf.reshape(model_pred, (seq_length, 9))
            if prediction[(idx + offset):].shape[0] < seq_length:
                pred_shape = prediction[(idx + offset):].shape[0]
                prediction[(idx + offset):] = model_pred[:pred_shape]
            else:
                prediction[(idx + offset):(idx + total_size)] = model_pred

        prediction = (prediction * mc_std) + mc_mean
        prediction = prediction[offset:]
        prediction = prediction.astype(np.int32)

        global_predictions.extend(prediction)

np.savetxt("mocap.csv", global_predictions, header=HEADER, delimiter=",", fmt="%i", comments="")
