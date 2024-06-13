import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import cv2 
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout 
import tensorflow as tf 
from tensorflow.keras.optimizers import SGD 
from keras import optimizers 
from keras.callbacks import ReduceLROnPlateau, EarlyStopping 
from tensorflow.keras.utils import to_categorical 
from keras.callbacks import EarlyStopping, ModelCheckpoint 

my_data = pd.read_csv('D:\Projects\DL FINAL\Data/A_Z Handwritten Data.csv').astype('float32')
my_frame = pd.DataFrame(my_data)

x = my_frame.drop('0', axis = 1)
y = my_frame['0']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))

print('Train Data Shape:', x_train.shape)
print('Test Data Shape:', x_test.shape)

word_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'
}

shuff = shuffle(x_train[:100])

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
print("New shape of train data:", x_train.shape)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
print("New shape of test data:", x_test.shape)

categorical_train = to_categorical(y_train, num_classes = 26)
print("New shape of train labels:", categorical_train.shape)

categorical_test = to_categorical(y_test, num_classes = 26)
print("New shape of test labels:", categorical_test.shape)

hcr_dl_model = Sequential()

hcr_dl_model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
hcr_dl_model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

hcr_dl_model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
hcr_dl_model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

hcr_dl_model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
hcr_dl_model.add(MaxPool2D(pool_size = (2, 2), strides = 2))

hcr_dl_model.add(Flatten())

hcr_dl_model.add(Dense(32, activation = "relu"))
hcr_dl_model.add(Dense(64, activation = "relu"))
hcr_dl_model.add(Dense(128, activation = "relu"))

hcr_dl_model.add(Dense(26, activation = "softmax"))

hcr_dl_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


history = hcr_dl_model.fit(x_train, categorical_train, epochs = 4, validation_data = (x_test, categorical_test))

hcr_dl_model.save('best_model.keras')

hcr_dl_model.evaluate(x_test,categorical_test)

hcr_dl_model.summary()

plt.figure(figsize = (6,6))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.title("Model Loss")
plt.show()

plt.figure(figsize = (6,6))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.title("Model Accuracy")
plt.show()
