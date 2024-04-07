
from tensorflow.keras.models import load_model
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
batch_size = 16

# Labels (assuming you've formatted your labels list correctly)
labels = [
    "Ah","Ou", "ba_htoat_chite", "ba_kone", "da_htway", "da_out_chite",
    "da_yay_hmote","da_yin_kout", "ga_khi", "ga_nge", "ha", "hsa_lain",
    "hta_hsin_htu", "hta_wun_beare", "ka_kji", "kha_khway", "la", "la_kji",
    "ma", "na_kji", "na_nge", "nga", "nya_kyi","pa_sout", "pfa_u_htoat",
    "sah_lone", "ta_wun_pu", "tha", "wa", "yah_kout",
    "yah_pet_let", "za_kwear", "za_myin_hsware"
]
num_classes = len(labels)

# Data Generators with Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
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



# If you saved the entire model
model = load_model('SQNet_finetuned1.h5')
print("model loaded")



# Example of SGD optimizer with momentum
optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.85)

# Learning Rate Scheduler
def lr_schedule(epoch, lr):
    # Reducing the learning rate by half every 5 epochs
    if epoch > 0 and epoch % 5 == 0:
        return lr * 0.5
    return lr

# Model compilation with the SGD optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Adding the Learning Rate Scheduler to the callbacks
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_schedule)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,  # You can adjust the number of epochs
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks  # Adding callbacks for learning rate adjustments
)

model.save_weights('SQNet_finetuned_weights2.h5')
# Or save the entire model
model.save('SQNet_finetuned2.h5')


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



