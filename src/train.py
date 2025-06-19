
import argparse
import os
import tensorflow as tf
import efficientnet.tfkeras as efn # EfficientNet models for Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Image data loading and augmentation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # Layers for building the model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau  # Training callbacks
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall

def model_fn(num_classes, image_size):
    
    # Load the EfficientNetB0 base model (without top classification layers), using pretrained ImageNet weights
    base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    
    # Freeze layers before a certain point (fine-tuning from 'block6a_expand_conv' and onward)
    # EfficientNetB0 has over 200 layers - before: block6a_expand_activation,  next block6a_expand_conv
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block6a_expand_conv':
            set_trainable = True
        layer.trainable = set_trainable

    # Build the full model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(), # Reduces parameters and helps prevent overfitting. Converts 3D output of CNN (e.g., (7, 7, 1280)) into 1D (e.g., (1280,)).
        Dense(512, activation='relu'), # ReLU activation introduces non-linearity.
        Dropout(0.2), # Randomly disables 20% of neurons during training.
        Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2)), # Uses LeakyReLU activation, which allows a small gradient
        Dropout(0.2),
        Dense(32, activation='relu'), # Another fully connected layer with fewer neurons (32), acting as a compression layer to reduce dimensionality.
        Dense(num_classes, activation='softmax') # softmax ensures the output is a probability distribution (all values sum to 1).
    ])
    
    return model

def main(args):
    train_dir = args.train
    val_dir = args.validation
    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epochs

    # Image augmentation for training data
    train_gen = ImageDataGenerator(
        rotation_range=15, # Randomly rotate images in the range (degrees, 0 to 15). Helps model become rotation-invariant.
        rescale=1./255, # Rescales pixel values from [0, 255] to [0, 1]. Normalization step.
        shear_range=0.2, # Applies random shearing (in radians). Distorts image diagonally to simulate perspective.
        zoom_range=0.2, # Randomly zooms in or out of the image by up to 20%. Helps model learn scale-invariance.
        horizontal_flip=True, # Randomly flips images horizontally. Useful for symmetrical classes (e.g., left vs right).
        fill_mode='nearest',  #Determines how to fill in newly created pixels after rotation/shear/zoom/shift. 'nearest' repeats closest pixel.
        width_shift_range=0.1, # Randomly shifts image width-wise by 10% of total width. Helps handle misalignments.
        height_shift_range=0.1,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    # Validation data (no augmentation, only normalization and preprocessing)
    val_gen = ImageDataGenerator(
        rescale=1./255, # Normalizes pixel values from [0-255] to [0-1]
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    # Create training data loader
    train_generator = train_gen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),  # Resize to match model input
        batch_size=batch_size, # Number of images to include in each batch
        class_mode='categorical',
        color_mode="rgb", # Color format of the images
        shuffle=True, # Randomly shuffle the order of images in each epoch
        seed=42 # Ensures the same random shuffling each time
    )

    # Create validation data loader
    val_generator = val_gen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Define performance metrics
    metrics = [
        CategoricalAccuracy(), Precision(), Recall()
    ]

    # Build model with appropriate number of output classes (e.g., Tomato - Bacterial Spot, Early Blight and etc)
    model = model_fn(num_classes=len(train_generator.class_indices), image_size=image_size)

    # Compile the model with Adam optimizer and cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    # Set up training callbacks
    callbacks = [
        #Minimum change in monitored quantity to qualify as an improvement. 
        ##Training stops if improvement is less than 0.00001
        ##Will wait 15 epochs before stopping if no improvement
        EarlyStopping(min_delta=1e-5, patience=15, restore_best_weights=True),
        ModelCheckpoint(os.path.join(args.model_dir, 'best_model.h5'), save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001)
    ]
    
    # Train the model
    model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks)

    # Save the final model
    model.save(os.path.join(args.model_dir, 'model.h5'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)
