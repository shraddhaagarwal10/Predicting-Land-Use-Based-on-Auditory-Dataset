'''
Author: - Abhishek Kuriyal

Model training script using keras module of the tensorflow library.

Training data extracted from ./image_datasets/train/
Validation data extracted from ./image_datasets/validation/

Images extracted from folders are completely random and are well shuffled.

Visualization done using Matplotlib at the end of the training.

Testing done on different script "model_prediction.py" to avoid cluttering.


'''

# Importing necessary libraries.

import tensorflow as tf
import glob
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import ssl
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
ssl._create_default_https_context = ssl._create_unverified_context

# Normalizing pixel values to be between 0 and 1


train = ImageDataGenerator(rescale=1/255.0)
validation = ImageDataGenerator(rescale=1/255.0)

# Extracting batches of training and validation data from respective folders.

train_set = train.flow_from_directory("image_datasets/train/", target_size=(64, 64), batch_size= 32, class_mode="categorical", shuffle = True, seed=42)

validation_set = validation.flow_from_directory("image_datasets/validation/", target_size=(64, 64), batch_size=32, class_mode="categorical", shuffle = True, seed=42)


# Building the model.

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(3, activation='softmax'))
model.summary()


# Compiling the model.

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(train_set, epochs=3, 
                    validation_data=validation_set)
                 

# Plotting the training curve.
                                 
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.plot(history.history['loss'], label = 'loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Model Evaluation
test_loss, test_acc = model.evaluate(validation_set, verbose=2)     


# Saving the model into model_save_state folder.
model.save("./model_save_state/model3")
   
