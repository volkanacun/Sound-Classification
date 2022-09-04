import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Load the data
dataset = pd.read_csv('THESIS_DATA_MASTER.csv')
print(len(dataset))
print(dataset.head())



###IMPORTANT####
# If you need to convert the data types use below
dataset.dtypes # check the data types
dataset['Overall'] = dataset.Flute.astype(int) # use this to change only the data type, but it wont work if there is infinite or NaN in the columns
dataset['Overall'] = dataset['Overall'].fillna(0).astype(np.int64) # Use this to get rid of infinite or NaN values and change the data type
dataset.dtypes # check the data types again
#####

# Define the features and the output
X = dataset.iloc[:, :-1].values    # we have 20 columns of data (so range = 0-19), this line slices the columns up to up. It does not take 19th row(which is actually the 20th)
y = dataset.iloc[:, -1].values     # this line only takes the 19th columns(which is actually the 20th)


#Label Encoder or OneHotEncoder (Used for country and languages here)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [9,10])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Check the shape
#y_train.head()
#print(x_train.shape)
#print(y_train.shape)


##### MODEL #####
# best accury is achived between 20-25 nodes 
# loss = sparse categorical corssentropy since the onehotencoder changed the values to float. Use categorical_crossentropy if its int. 

model = tf.keras.models.Sequential()
# input layer
model.add(tf.keras.layers.Dense(units=17, kernel_initializer='uniform', activation='relu'))
# 1st hidden layer
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
 # 1st hidden layer
model.add(tf.keras.layers.Dense(units=10, activation='relu'))
# 3rd hidden layer
model.add(tf.keras.layers.Dense(units=10, activation='relu'))


# Output layer
model.add(tf.keras.layers.Dense(6, activation='softmax'))

#adam = tf.keras.optimizers.Adam(lr=0.0001)
sgd = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=80, epochs=150, validation_data=(X_test, y_test))
model.summary()


####### METRICS PLOTS #######
print(history.history.keys())

# Visualize Loss 
# Get training and test loss histories

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy (100 epoch, 3 nodes)')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
# Visualize Loss 2.0
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss (100 epoch, 3 nodes))')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Confusion Matrix
# Predict the test set results
y_pred = model.predict(X_test)
y_pred

# Evaluate Model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)
print(f1_score(y_test, y_pred, average='micro'))
print(accuracy_score(y_test,y_pred))


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)