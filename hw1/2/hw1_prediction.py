from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
import keras.backend.tensorflow_backend as K
import numpy

print(device_lib.list_local_devices())

# random seed for reproducibility
numpy.random.seed(2)

# loading preprocessed csv 
dataset = numpy.loadtxt("Churn_Modelling_Preprocessed.csv", delimiter=",")

# input x outpuy y
X = dataset[:,0:10]
Y = dataset[:,10]

# 80:20 train and test 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#using gpu
with K.tf.device('/gpu:0'):
    # create model
    model = Sequential()
    model.add(Dense(15, input_dim=10, activation='relu')) # input layer requires input_dim param
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
   
    # train
    model.fit(x_train, y_train, epochs = 10, batch_size=5, validation_data=(x_test, y_test))

    # predict
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5)

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy = 1.0*(cm[0,0] + cm[1,1]) / (cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    print("\nAccuracy: %.2f%%" % (accuracy))