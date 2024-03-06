import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define directories containing cat and non-cat images
train_dir = './train'
validation_dir = './train'

# Define image dimensions and batch size
img_width, img_height = 150, 150
batch_size = 32

# Use ImageDataGenerator to preprocess and load images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Check if trained model exists
try:
    # Load the saved model
    model = load_model('cat_classifier.h5')
    print("Trained model loaded successfully.")
except:
    # Define the CNN model if the saved model doesn't exist
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )

    # Save the trained model
    model.save('cat_classifier.h5')
    print("Trained model saved successfully.")

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print('Validation Loss:', loss)
print('Validation Accuracy:', accuracy)

import numpy as np
from tensorflow.keras.preprocessing import image

# Load a sample image
img_path = 'cat.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
if prediction[0] < 0.5:
    print('Not a cat')
else:
    print('Cat')
