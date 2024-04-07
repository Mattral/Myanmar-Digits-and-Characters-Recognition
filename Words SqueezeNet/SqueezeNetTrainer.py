import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Directories
train_dir = 'data/'
validation_dir = 'validate/'



# Basic Parameters
img_height, img_width = 224, 224
batch_size = 32

# Labels (assuming you've formatted your labels list correctly)
labels = [
    "Ah", "Ou", "ba_htoat_chite", "ba_kone", "da_htway", "da_out_chite",
    "da_yay_hmote","da_yin_kout", "ga_khi", "ga_nge", "ha", "hsa_lain",
    "hta_hsin_htu", "hta_wun_beare", "ka_kji", "kha_khway", "la", "la_kji",
    "ma", "na_kji", "na_nge", "nga", "nya_kyi","pa_sout", "pfa_u_htoat",
    "sah_lone", "ta_wun_pu", "tha", "wa", "yah_kout",
    "yah_pet_let", "za_kwear", "za_myin_hsware"
]
num_classes = len(labels)

print(num_classes)

# Data Generators with Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    shear_range=0.1,  
    zoom_range=0.1,  
    fill_mode='nearest'
)


validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)


#complete SqueezeNet
def fire_module(x, squeeze_filters, expand_filters, l2_reg=0.001):
    squeezed = layers.Conv2D(squeeze_filters, (1, 1), activation='relu', padding='same',
                             kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    e1x1 = layers.Conv2D(expand_filters, (1, 1), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(squeezed)
    e3x3 = layers.Conv2D(expand_filters, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(squeezed)
    x = layers.Concatenate()([e1x1, e3x3])
    return x

def SqueezeNet(input_shape=(224, 224, 3), num_classes=len(labels)):
    input_image = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_image)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)

    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Activation('softmax')(x)
    
    model = models.Model(inputs=input_image, outputs=output)
    return model

# Adjust the num_classes based on your dataset
num_classes = len(labels)
model = SqueezeNet(input_shape=(224, 224, 3), num_classes=num_classes)

# Example of SGD optimizer with momentum
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.85)

# Learning Rate Scheduler
def lr_schedule(epoch, learning_rate):
    # Reducing the learning rate by half every 5 epochs
    if epoch > 0 and epoch % 4 == 0:
        return learning_rate * 0.5
    return learning_rate

import math
steps_per_epoch = math.ceil(train_generator.samples / batch_size)
validation_steps = math.ceil(validation_generator.samples / batch_size)

print("step per epoch : val steps")
print(steps_per_epoch)
print(validation_steps)

# Model compilation with the SGD optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Adding the Learning Rate Scheduler to the callbacks
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_schedule)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks  # Adding callbacks for learning rate adjustments
)


# Save model weights
model.save_weights('SQNet_weights.h5')

# Save the entire model
model.save('SQNet.h5')


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

print(validation_generator.class_indices)

# Predict the validation dataset
validation_generator.reset()  # Resetting generator to ensure proper class indices
predictions = model.predict(validation_generator, steps=validation_generator.samples // validation_generator.batch_size+1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# Compute confusion matrix
cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(class_labels)))

# Plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


