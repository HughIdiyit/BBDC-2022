# BBDC 2022: A guide for executing the code of the team Kornstante
* Firstly, the requirements given in `requirements.txt` have to be installed.
* The data given in the challenge has to be located in a data directory within the root directory.

## General approach
* Mocap: Was solved using a one layer LSTM, with a history of 105 and prediction length of one frame.
* Video: Was treated as an inbetweening problem. Two convolutional LSTMs were trained. One predicts in the forward direction, the other one predicts in the backward direction.
  * ConvLSTM forward: Takes 15 frames in front of a given gap as it's history and predicts the following 5 frames.
  * ConvLSTM backward: Firstly, the data is reversed along the time axis. The network takes 15 frames after a gap as it's history and predicts the 5 frames in front of the history frames.
* The mocap and video prediction are carried out using a sliding window over the missing frames.

## Automatische Ausführung der Prediction
* Um alle Ergebnisse automatisch zu erstellen muss nur die Datei `prediction.py` ausgeführt werden.
* In dieser sind bereits die trainierten Modelle ausgewählt und alle Einstellungen vorgenommen.
* Zur Reproduktion unserer Daten reicht daher die Ausführung dieses Skripts.

## Ausführung des Trainings
1. Ausführung der Datei `preprocessing.py`
    * Die gewünschten Sequenzlängen, sowohl für die Eingabe als auch für das Label, werden als Kommandozeilen Argumente
an die Methoden übergeben.
2. Das Training der Mocap Daten befindet sich in `mocap_train.py`
    * In dieser Datei befinden sich globale Variablen (komplett in Großbuchstaben geschrieben), in denen das Training
konfiguriert werden kann.
    * Das trainierte Model wird letztlich automatisch in den Ordner `best_model_mocap` exportiert.
3. Training der Video Prediction
    * Die Video Prediction erfolgt zweiseitig auf den Daten.
    * Training des "vorwärts"-Models mittels `video_train.py`.
    * Die "Rückrichtung" wird in `reverse_video_train.py` trainiert.
    * Auch hier wird in beiden Dateien das trainierte Model automatisch exportiert.
      * In `torch_models/convlstm/` bzw. `torch_models/reverse_convlstm/`

## Executing the prediction
* Run `preprocessing.py`
### Mocap Prediction
1. Run `mocap_prediction.py`
2. Run `mocap_postprocessing.py`
### Video Prediction
1. Run `vid_pred_twosided.py`
