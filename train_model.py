import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Image size
img_size = 128

# Load dataset
datagen = ImageDataGenerator(
    rescale=1/255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(img_size, img_size),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# Create simple CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="softmax"))

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
model.fit(train_data, validation_data=val_data, epochs=5)

# Save model
model.save("vitamin_model.h5")

print("Model trained and saved successfully!")
