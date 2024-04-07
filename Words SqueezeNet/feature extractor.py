
'''
# Labels (Ensure these are in the same order as the folders if sorting alphabetically)
labels = [
    "Ah", "Ou", "ba_htoat_chite", "ba_kone", "da_htway", "da_out_chite",
    "da_yay_hmote", "da_yin_kout", "ga_khi", "ga_nge", "ha", "hsa_lain",
    "hta_hsin_htu", "hta_wun_beare", "ka_kji", "kha_khway", "la", "la_kji",
    "ma", "na_kji", "na_nge", "nga", "nya_kyi", "pa_sout", "pfa_u_htoat",
    "sah_lone", "ta_wun_pu", "tha", "wa", "yah_kout",
    "yah_pet_let", "za_kwear", "za_myin_hsware"
]
'''

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

def extract_features_and_labels(generator, model, labels, batch_size=32):
    """
    Generator function to extract features and labels
    and immediately save them to a CSV file in batches.
    """
    batch_features = []
    batch_labels = []

    for i in range(len(generator)):
        img, label_index = generator.next()
        feature = model.predict(img)
        batch_features.append(feature.flatten())
        batch_labels.append(labels[int(label_index)])
        
        # When the batch is full, yield the current batch
        if len(batch_features) >= batch_size:
            yield np.array(batch_features), np.array(batch_labels)
            batch_features = []
            batch_labels = []

    # Yield any remaining features and labels as the last batch
    if batch_features:
        yield np.array(batch_features), np.array(batch_labels)

def save_features_to_csv(features_generator, csv_path):
    """
    Save features to a CSV file in chunks to reduce memory usage.
    """
    header = True  # Only write the header once
    for features, labels in features_generator:
        df = pd.DataFrame(features)
        df['label'] = labels
        df.to_csv(csv_path, mode='a', index=False, header=header)
        header = False  # Disable the header after the first write

# Directory where your images are stored
base_dir = 'validate/'

# Initialize the ResNet50 model
model = ResNet50(include_top=False, input_shape=(224, 224, 3), pooling='avg', weights='imagenet')

# Data generator
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(base_dir, target_size=(224, 224), batch_size=1, class_mode='sparse', shuffle=False)

# Extract features and labels using a generator
features_generator = extract_features_and_labels(generator, model, labels, batch_size=10)

# Save features and labels to CSV
csv_path = 'image_features.csv'
save_features_to_csv(features_generator, csv_path)

print("Features have been successfully saved to", csv_path)


""" Example for infernece:
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load and preprocess an image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Example usage
img_path = 'path/to/your/image.jpg'
prepared_img = prepare_image(img_path)

# Assuming 'model' is your trained model (or the ResNet50 feature extractor)
predictions = model.predict(prepared_img)

# Post-processing the prediction
# This step will vary. Here's how to decode predictions if you're using the full ResNet50 model
# For feature extraction, you might directly use the features or apply a separate classifier
# decoded_predictions = decode_predictions(predictions, top=3)[0]
# print("Predictions:", decoded_predictions)

# For a feature extraction model, you'd typically follow with a classification step here

"""
