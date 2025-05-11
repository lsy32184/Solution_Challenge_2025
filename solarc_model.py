import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# Training Round File Path
ROUND_FILE_PATH = 'checkpoints/training_round.txt'

# Checkpoint Path with Training Round
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Determine Training Round
if os.path.exists(ROUND_FILE_PATH):
    with open(ROUND_FILE_PATH, 'r') as f:
        TRAINING_ROUND = int(f.read().strip()) + 1
else:
    TRAINING_ROUND = 1

# Save updated training round
with open(ROUND_FILE_PATH, 'w') as f:
    f.write(str(TRAINING_ROUND))

# Checkpoint Path
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f'best_model_round_{TRAINING_ROUND}.h5')

print(f"Starting Training Round {TRAINING_ROUND}")
print(f"Checkpoint Path: {CHECKPOINT_PATH}")

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

# Train/Validation/Test Split (일관성 있는 데이터 분할)
train_metadata = metadata.sample(frac=0.8, random_state=42)
temp_metadata = metadata.drop(train_metadata.index)
valid_metadata = temp_metadata.sample(frac=0.5, random_state=42)
test_metadata = temp_metadata.drop(valid_metadata.index)

print(f"Training Set: {len(train_metadata)}, Validation Set: {len(valid_metadata)}, Test Set: {len(test_metadata)}")

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

# Training Generator
train_generator = train_datagen.flow_from_dataframe(
    train_metadata,
    x_col='image_path',
    y_col='dx',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

# Model Architecture
base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
num_classes = len(train_generator.class_indices)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Checkpoint Callback
checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor='loss',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint_callback]
)

print(f"Training complete for Round {TRAINING_ROUND}. Model saved at {CHECKPOINT_PATH}")
