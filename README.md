Recognizing handwritten characters in english using a deep learning approach. The dataset used is a csv file (~600MB) that contains grayscale values of english alphabets with 26 classes(duh)

Dataset used - https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format

The 'model-building.py' is where the model is built and saved as 'best_model.keras', I have used three convolution layers with relu activation function, three dense layers with relu and one final dense layer with softmax activation function

In 'hcr.py' the saved model is loaded in and and the dataset is split into test set and shuffled, prediction is done and plotted.

The 'Dataset-Sample.py' does what it says, since I made this project in VScode it seemed rather inconvenient to run the file just to get a sample of the dataset, especially because the dataset is unreadeable in a convenient manner.

![dataset sample](https://github.com/ambareeshav/HCR/assets/126247692/066b4297-9a27-431c-9c7a-77a1f6c71d87)


The best I was able to achieve after some trial and error with layers and filter sizes was a accuracy of 0.987 and a loss of 0.0754, 
since this was only an introduction to the vsat field of deep learning and AI I have not spent much time tweaking parameters for better accuracy and loss.

![model loss3](https://github.com/ambareeshav/HCR/assets/126247692/379958e6-23ba-4333-86b5-f461b766f385) 
![model accuracy3](https://github.com/ambareeshav/HCR/assets/126247692/d723e638-5973-4895-9631-a9b6629de816)

