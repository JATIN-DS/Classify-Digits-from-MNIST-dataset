
# In[1]:
import numpy
import keras
from keras import backend as k
from keras.datasets import mnist
from keras.utils import np_utils   # This function takes a vector or 1 column matrix of class labels and converts it into a matrix with p columns, one for each category.


# In[2]:


from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
import pandas as pd


# In[3]:


model = Sequential()


# In[4]:


'''
Both TensorFlow and Theano expects 4D tensors of image data as input. But, while TensorFlow expects 
its structure/shape to be (samples, rows, cols, channels), Theano expects it to be (samples, channels, rows, cols). 
So, setting the image_dim_ordering to 'tf' made Keras use the TensorFlow ordering, while setting it to 'th' made it 
Theano ordering.
'''
# image_data_format can be set to "channels_last" or "channels_first", 
# which corresponds to the TensorFlow or Theano dimension orders respectively.
k.set_image_data_format('channels_last')
numpy.random.seed(0)


# In[5]:


path_train = r'C:\Users\Jatin Mittal\Downloads\python codes for deep learning\CNN\MNIST dataset\train.csv'
path_test = r'C:\Users\Jatin Mittal\Downloads\python codes for deep learning\CNN\MNIST dataset\test.csv'

X = pd.read_csv(path_train)
test = pd.read_csv(path_test)

y = X['label']
X.drop(['label'],inplace=True, axis=1)


# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2 , random_state=42)


# In[7]:


'''
We then reshape the samples according to TensorFlow convention which we chosed previously using
"K.set_image_data_format('channels_last')" samples,rows,columns,channels as we are using channels_last if you
are using channels_first you will need to change the order to samples,channels,rows,column and here we have only one
channel because we are using the image in grayscale not RGB. 
'''
'''
NOTE: The data we have is in integer form i.e. integer values behind the images. So we need to reshape that flatten data
in the form that that flatten data now reshape represents 1 single image. 

If the input data are images itself and not the values then python has a solution for that, it used flow_from_directory()
function that automatically reads all images as data.
'''

X_train = X_train.reshape(X_train.shape[0], 28, 28 , 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28 , 1).astype('float32')


# In[9]:


# data exploration

import matplotlib.pyplot as plt
print("the number of training examples = %i" % X_train.shape[0])
print ("the number of classes: %i" % len(numpy.unique(y_train)))
print("Dimention of images = {} x {}  ".format(X_train.shape[1],X_train.shape[2])  )

unique, count = numpy.unique(y_train, return_counts=True)
print("the number of unique classses are and it's count is: %s" %dict(zip(unique, count)))

images_labels = list(zip(X_train, y_train))

for index, (images, labels) in enumerate (images_labels[:12]):
    plt.subplot(4,3, index+1)         # 4 is number of subplots in y direction and 3 is  in x direction, index+1 is total subplots
    plt.axis('off')
    plt.imshow(images.squeeze(), cmap=plt.cm.gray_r, interpolation='nearest') # squeeze temporarily deletes last dimension i.e. (28x28x1) is now (28x28) 
    plt.title('label: %i' % labels )


# In[8]:


from keras.layers import Dropout

model.add(Conv2D(40, kernel_size = 5, padding = "same", input_shape=(28,28,1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(Conv2D(500, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(1024, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


# In[10]:


from keras.layers.core import Activation

model.add(Flatten())
model.add(Dense(units=100, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(units=50, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=70, activation="relu"))
model.add(Dropout(0.1))
model.add(Dense(units=30, activation="relu"))
model.add(Dropout(0.4))

model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])


# In[11]:


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# In[ ]:


# now we will generate new data for the input images

from keras.preprocessing.image import ImageDataGenerator

X_train2 = numpy.copy(X_train)
y_train2 = numpy.copy(y_train)

datagen = ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20)

datagen.fit(X_train)

result_x = numpy.concatenate((X_train, X_train2), axis=0)
result_y = numpy.concatenate((y_train, y_train2), axis=0)

training_set = datagen.flow(result_x, result_y, batch_size= 35)

# now fit the model to training set
history = model.fit_generator(training_set, steps_per_epoch=400, epochs = 2)


# In[ ]:


scores = model.evaluate(X_test, y_test, verbose = 10)
print(scores)


# In[ ]:


test_set = test_set.reshape(-1, 28,28,1)

result = model.predict(test_set)
res = numpy.argmax(res,axis = 1)      # note: the ouput is in form [1,0,0,0,0,0,0,0] or [0,0,0,1,0,0,0...], so argmax gives us index of position where 1 occurs which is actually predicted value
res = pd.Series(res, name="Label")

# submitting result in prescribed format
submission = pd.concat([pd.Series(range(1 ,28001) ,name = "ImageId"),   res],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)
submission.head(10)

