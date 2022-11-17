# BBDC 2022: A guide for executing the code of the team Kornstante
* Firstly, the requirements given in `requirements.txt` have to be installed.
* The data given in the challenge has to be located in a data directory within the root directory.

## General approach
* Mocap: Was solved using a one layer LSTM, with a history of 105 and prediction length of one frame.
* Video: Was treated as an inbetweening problem. Two convolutional LSTMs were trained. One predicts in the forward direction, the other one predicts in the backward direction.
  * ConvLSTM forward: Takes 15 frames in front of a given gap as it's history and predicts the following 5 frames.
  * ConvLSTM backward: Firstly, the data is reversed along the time axis. The network takes 15 frames after a gap as it's history and predicts the 5 frames in front of the history frames.
* The mocap and video prediction are carried out using a sliding window over the missing frames.

## Training the models
1. Run the preprocessing script `preprocessing.py`
    * The sequence lengths for the model input and label are passed as command-line arguments
    * The arguments are parsed and passed to the function calls
2. The mocap model is trained in the `mocap_train.py` file
    * The training is configured via global variables in the file, e.g. *BS* for the batch size
    * The trained model is saved to the `best_model_mocap` folder
3. Training the video prediction model
    * The video prediction is a bi-directional temporal reconstruction
    * The "forward" model is trained in `video_train.py`
    * The "reverse" model is trained via `reverse_video_train.py`
    * Both trained models are saved to their respective folders
        * `torch_models/convlstm/` and `torch_models/reverse_convlstm/`

## Executing the prediction
* Run `preprocessing.py`
### Mocap Prediction
1. Run `mocap_prediction.py`
2. Run `mocap_postprocessing.py`
### Video Prediction
1. Run `vid_pred_twosided.py`
