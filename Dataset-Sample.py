import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # type: ignore
import cv2 # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.utils import shuffle # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from keras.models import load_model # type: ignore

my_data = pd.read_csv('D:\Projects\DL FINAL\Data/A_Z Handwritten Data.csv').astype('float32')
my_frame = pd.DataFrame(my_data)

x = my_frame.drop('0', axis = 1)
y = my_frame['0']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))

shuff = shuffle(x_train[:100])


fig, ax = plt.subplots(5, 5, figsize = (10, 10))

axes = ax.flatten()

for i in range(25):
    shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i], (28, 28)), cmap = 'Greys')
plt.show()