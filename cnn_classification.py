from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

train_dir = r'D:\study\毕设\cnn\dataset\train'
validation_dir = r'D:\study\毕设\cnn\dataset\validation'


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150,3)))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu',name='dense_1'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(180, activation='softmax',name='dense_2'))


model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])



# 调整像素值
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    )
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(150, 150),
    batch_size=192,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(150, 150),
    batch_size=128,
    class_mode='categorical')
#print(train_generator.class_indices)

model_checkpoint = ModelCheckpoint('categorical_model.h5', 
                                   monitor='loss',
                                   verbose=1, 
                                   save_best_only=False,
                                   period=5,
                                   save_weights_only=False)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=60,
    epochs=100,
    callbacks=[model_checkpoint],
    validation_data=validation_generator,
    validation_steps=30)



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
