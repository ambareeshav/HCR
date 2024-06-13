import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import cv2 
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 
from tensorflow.keras.utils import to_categorical 
from keras.models import load_model 

my_data = pd.read_csv('D:\Projects\DL FINAL\Data/A_Z Handwritten Data.csv').astype('float32')
my_frame = pd.DataFrame(my_data)

x = my_frame.drop('0', axis = 1)
y = my_frame['0']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
x_train = np.reshape(x_train.values, (x_train.shape[0], 28, 28))
x_test = np.reshape(x_test.values, (x_test.shape[0], 28, 28))

word_dict = {
    0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'
}

shuff = shuffle(x_train[:100])

categorical_test = to_categorical(y_test, num_classes = 26)

hcr_dl_model = load_model('best_model.keras')

fig, axes = plt.subplots(5, 5, figsize = (15, 15))
axes = axes.flatten()

for i, ax in enumerate(axes):
    img = np.reshape(x_test[i], (28, 28))
    ax.imshow(img, cmap = 'Greys')
    
    pred = word_dict[np.argmax(categorical_test[i])]
    ax.set_title(pred, fontsize = 20, fontweight = 'bold', color = 'blue')
    ax.grid()
plt.show()