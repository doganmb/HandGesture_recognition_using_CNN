import tensorflow as tf
from keras.models import Sequential,Model,model_from_json
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation,Dropout,Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import numpy as np


data = np.loadtxt("./CSV/new_data.csv",dtype=int)

x_data = data[:,0:-1]
y_data = data[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.2)

x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

train_label_flat = y_train.ravel()
train_label_count = np.unique(train_label_flat).shape[0]

def dense_to_one_hot(label_dense,num_classes):
    num_labels = label_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + label_dense.ravel()] = 1
    return labels_one_hot

y_train = dense_to_one_hot(train_label_flat,train_label_count)
y_train = y_train.astype(np.uint8)

test_labels_flat = y_test.ravel()
test_labels_count = np.unique(test_labels_flat).shape[0]

y_test = dense_to_one_hot(test_labels_flat,test_labels_count)
y_test = y_test.astype(np.uint8)

x_train = x_train.reshape(-1,70,70,1)
x_test = x_test.reshape(-1,70,70,1)

"""
datagen = ImageDataGenerator(
    brightness_range = (0.3,0.7))

datagen.fit(x_train)
"""
model = Sequential()

model.add(Conv2D(32,kernel_size= (3,3) ,activation='relu',input_shape = (70,70,1)))
model.add(Conv2D(32,kernel_size= (3,3) ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(32,kernel_size= (3,3) ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Conv2D(32,kernel_size= (3,3) ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(6))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
model.summary()
checkpointer = ModelCheckpoint(filepath='new7_model_save.h5',verbose=1,save_best_only=True)
#predict = model.fit_generator(datagen.flow(x_train, y_train.toarray(), batch_size=100),epochs=7,batch_size=100,validation_data=(x_test,y_test),callbacks=[checkpointer],verbose=2)
predict = model.fit(x_train,y_train,epochs=7,batch_size=100,validation_data=(x_test,y_test),callbacks=[checkpointer],verbose=2)
