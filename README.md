# An Approach Based on Sound Classification to Predict Soundscape Perception Through Machine Learning 

A PhD thesis by Volkan Acun, carried out at Bilkent University between 2018 and 2021 for the Department of Interior Architecture and Environmental Design

## Content of the Repository

The repository includes four python modules for a Convolutional Neural Network based sound classification model:
* __init__.py
* feature_extract.py
* models.py
* train_pred.py

and it also contains a Feedforward Neural Network
* ffnn.py

The CNN model is based on a musical information retrieval network, which we use to classify the audio content of environmental sound recordings. Tuning the hyperparameters,
training the network and predictions on a new audio file are performed by using the train_pred.py file. The audio content output from the CNN model is combined with the 
auditory perception data gathered through a questionnaire survey. The "ISO/TS 12913-2:2018 Acoustics — Soundscape — Part 2: Data collection and reporting requirements(ISO, 2018) standard
is used as a basis for the questionnaire survey. ffnn.py is used for analysing the combined data file and making predictions about individuals' overall response to different soundscapes

## References
<a id="1"></a> 
International Organization for Standardization. (2018). Acoustics — Soundscape — Part 2: Data collection and reporting requirements (ISO Standard NO. 12913-2:2018). https://www.iso.org/standard/75267.html
