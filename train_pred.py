
#%%
#  make a prediction

import numpy as np
from pyaudioclassification import feature_extraction, train, predict, print_leaderboard

parent_dir = 'e:\Machine Learning\Pro_03_env'
#%%
# step 1: preprocessing
if np.DataSource().exists("./feat_G04_FINAL.npy") and np.DataSource().exists("./label_G04_FINAL.npy"):
    features, labels = np.load('./feat_G04_FINAL.npy'), np.load('./label_G04_FINAL.npy')
else:
    features, labels = feature_extraction('./data/')
    np.save('./feat_G04_FINAL.npy', features)
    np.save('./label_G04_FINAL.npy', labels)

# step 2: training
if np.DataSource().exists("./model_G04_FINAL.h5"):
    from tensorflow.keras.models import load_model 
    model = load_model('./model_G04_FINAL.h5')
else:
    model = train(features, labels, type='cnn', print_summary=True, lr=0.03, loss_type='categorical',
             epochs=200, optimizer='SGD')
    model.save('./model_G04_FINAL.h5', include_optimizer=False)#include_optimzer wariod the warning:tensorlow error

# step 3: prediction
# %%
pred = predict(model, './Prediction/Survey_G03_Pred/371059__joshuaempyre__duduk-with-orchestra.wav')
print_leaderboard(pred, './data/')



