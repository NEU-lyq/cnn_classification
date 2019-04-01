from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import shutil
from PIL import Image

train_dir = r'D:\study\毕设\cnn\all - 副本'
validation_dir = r'D:\study\毕设\cnn\redataset\validation'


model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(150, 150,3)))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))


model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu',name='dense_1'))

model.add(layers.Dropout(0.3))
model.add(layers.Dense(2,name='dense_2'))

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['acc'])


lable = numpy.loadtxt(r'D:/study/毕设/cnn/lable.txt')
xpath=r'D:\study\毕设\cnn\all - 副本'
filelist = os.listdir(xpath) 
images=[]
labels=lable
for file in filelist:
    filename=os.path.splitext(file)[0]
    path=os.path.join(xpath,file)
    img=Image.open(path)
    img = img.resize((150, 150),Image.ANTIALIAS)
    img=np.array(img)
    images.append(img)
def train_gen():
    while True:
        batch_images=[]
        batch_labels=[]
        for i in range(192):
            if(i==len(images)):
                i=0
            batch_images.append(images[i])
            batch_labels.append(labels[i])
            i+=1
        batch_images=np.array(batch_images)
        batch_labels=np.array(batch_labels)
        yield batch_images,batch_labels

# 调整像素值
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=0,
#    width_shift_range=0,
#    height_shift_range=0,
#    shear_range=0,
#    zoom_range=0,
#    )

#validation_datagen = ImageDataGenerator(rescale=1./255)

#train_generator = train_datagen.flow_from_directory(
#    directory=train_dir,
#    target_size=(150, 150),
#    batch_size=192,
#
#    class_mode='sparse')

#validation_generator = validation_datagen.flow_from_directory(
#    directory=validation_dir,
#    target_size=(150, 150),
#    batch_size=128,
#    class_mode='sparse')

model_checkpoint = ModelCheckpoint('regression_model.h5', 
                                   monitor='loss',
                                   verbose=1, 
                                   save_best_only=False,
                                   period=1,
                                   save_weights_only=False)
history = model.fit_generator(
    generator=train_gen(),
    steps_per_epoch=100,
    epochs=10,
    callbacks=[model_checkpoint],
#    validation_data=validation_generator,
#    validation_steps=3
)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
