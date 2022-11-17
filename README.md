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

## Ausführung der Prediction
* Ausführung von `preprocessing.py`
### Mocap Prediction
1. Ausführung von `mocap_prediction.py`
2. Postprocessing der Daten durch Ausführung von `mocap_postprocessing.py`
### Video Prediction
1. Ausführung von `vid_pred_twosided.py`
