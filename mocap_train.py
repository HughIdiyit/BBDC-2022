import pandas as pd
import numpy as np
import tensorflow as tf

from data_preparation import WindowGenerator

BS = 128
LR = 0.01
EPOCHS = 10
UNITS = 32
HISTORY = 105
PRED_LEN = 1


def compile_and_fit(model, window):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint("./best_model_mocap",
                                                                   monitor='val_loss', mode='min', save_best_only=True)

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=LR),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=EPOCHS,
                        validation_data=window.val,
                        callbacks=[model_checkpoint_callback])
    return history


if __name__ == "__main__":
    val_df = pd.read_csv("data/s070t04_mocap_norm.csv")
    val_df.drop(list(val_df.filter(regex='Unnamed')), axis=1, inplace=True)

    keys = [k for k in val_df.keys()]
    all_samples = np.load("all_mocap_samples.npy")
    dataset = pd.DataFrame({h: all_samples[:, i] for i, h in enumerate(keys)})

    train_df = dataset.iloc[:int(0.9*len(dataset))]
    val_df = dataset.iloc[int(0.9*len(dataset)):]
    test_df = dataset.iloc[int(0.9*len(dataset)):]

    run_window = WindowGenerator(
        HISTORY, PRED_LEN, PRED_LEN, train_df, val_df, test_df, BS,
        label_columns=[k for k in keys])

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(UNITS, return_sequences=False),
        tf.keras.layers.LayerNormalization(),
        # Idea to include Dense layer in this manner taken from:
        # https://www.tensorflow.org/tutorials/structured_data/time_series
        tf.keras.layers.Dense(PRED_LEN * 9,
                              kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([PRED_LEN, 9])
    ])

    compile_and_fit(model, run_window)
