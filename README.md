# Anleitung zur Ausführung des Codes der Gruppe Kornstante
* Diese Anleitung ist aufgeteilt in die Ausführung unseres Trainings, falls erwünscht, 
und der Prediction mit unserem bereits trainierten Model.
* Zuerst müssen die Requirements aus `requirements.txt` installiert werden.
* Der data-Ordner muss in derselben Directory liegen, wie die mitgelieferten Dateien.

## General approach
* Mocap: Wurde mit einem 1-Layer LSTM gelöst, mit einer History von 105 und einer Vorhersage von einem Frame.
* Video: Wurde als inbetweening-Problem behandelt. Es wurden zwei ConvLSTMs trainiert. Eines arbeitet sich vorwärts,
eines rückwärts durch die Daten.
  * ConvLSTM forward: Nimmt sich 15 Frames vor einer Lücke als History und predicted die 5 folgenden Frames.
  * ConvLSTM backward: Zunächst werden die Daten entlang der Zeitachse umgekehrt. Das Netz nimmt sich 15 Frames nach
einer Lücke als "History" und predicted jeweils 5 zurückliegende Frames.
* Sowohl Mocap- als auch Video-Prediction werden mit einem sliding window über die zu füllenden Frames angewendet.

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

## Ausführung der Prediction
* Ausführung von `preprocessing.py`
### Mocap Prediction
1. Ausführung von `mocap_prediction.py`
2. Postprocessing der Daten durch Ausführung von `mocap_postprocessing.py`
### Video Prediction
1. Ausführung von `vid_pred_twosided.py`
