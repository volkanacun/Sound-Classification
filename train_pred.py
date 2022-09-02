
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


# %%
history = model.fit(X_train,y_train,batch_size=100, epochs=100, validation_data=(X_test, y_test))
model.summary()


####### METRICS PLOTS #######
print(history.history.keys())

#Evaluate Model
cm = confusion_matrix(y_test, pred)
print (cm)
print(f1_score(y_test, pred, average='micro'))
print(accuracy_score(y_test,y_pred))
print(roc_auc_score(y_test, y_pred))



# Visualize Loss 
# Get training and test loss histories
import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy (300 epoch, 4x10 nodes)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# Visualize Loss 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (300 epoch, 4x10 nodes)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


