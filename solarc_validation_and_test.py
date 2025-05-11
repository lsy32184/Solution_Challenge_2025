import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 7  # 클래스 개수 (dx 열의 고유값 수)
CHECKPOINT_PATH = 'checkpoints/best_model.h5'

# Data Paths
BASE_DIR = './'
IMAGE_DIR_1 = os.path.join(BASE_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_2 = os.path.join(BASE_DIR, 'HAM10000_images_part_2')
METADATA_PATH = os.path.join(BASE_DIR, 'HAM10000_metadata.csv')

# Load Metadata
metadata = pd.read_csv(METADATA_PATH)
metadata['image_path'] = metadata['image_id'].apply(
    lambda x: os.path.join(IMAGE_DIR_1, x + '.jpg') if os.path.exists(os.path.join(IMAGE_DIR_1, x + '.jpg')) 
    else os.path.join(IMAGE_DIR_2, x + '.jpg')
)

# Train/Validation/Test Split
train_metadata = metadata.sample(frac=0.8, random_state=42)
temp_metadata = metadata.drop(train_metadata.index)
valid_metadata = temp_metadata.sample(frac=0.5, random_state=42)
test_metadata = temp_metadata.drop(valid_metadata.index)

print(f"Validation Set: {len(valid_metadata)}, Test Set: {len(test_metadata)}")

# Data Augmentation (Only Normalization for Validation and Test)
datagen = ImageDataGenerator(rescale=1./255)

# Validation Generator
valid_generator = datagen.flow_from_dataframe(
    valid_metadata,
    x_col='image_path',
    y_col='dx',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Test Generator
test_generator = datagen.flow_from_dataframe(
    test_metadata,
    x_col='image_path',
    y_col='dx',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Define Model Architecture
def create_model(num_classes):
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create model and load weights
try:
    model = create_model(NUM_CLASSES)
    model.load_weights(CHECKPOINT_PATH)
    print(f"Model weights successfully loaded from {CHECKPOINT_PATH}")
except Exception as e:
    print(f"Error loading weights: {e}")
    model = None

# Validation Evaluation
if model:
    try:
        val_loss, val_acc = model.evaluate(valid_generator)
        print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
    except Exception as e:
        print(f"Error during validation: {e}")

# Testing Evaluation
if model:
    try:
        test_loss, test_acc = model.evaluate(test_generator)
        print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    except Exception as e:
        print(f"Error during testing: {e}")
