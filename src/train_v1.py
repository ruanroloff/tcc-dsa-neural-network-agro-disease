
import argparse
import os
import tensorflow as tf
#import efficientnet.tfkeras as efn
#replaced by keras - directly
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall

def model_fn(num_classes, image_size):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'block6a_expand_conv':
            set_trainable = True
        layer.trainable = set_trainable

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.2)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def main(args):
    train_dir = args.train
    val_dir = args.validation
    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epochs

    train_gen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        width_shift_range=0.1,
        height_shift_range=0.1,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    val_gen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    train_generator = train_gen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode="rgb",
        shuffle=True,
        seed=42
    )

    val_generator = val_gen.flow_from_directory(
        val_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    metrics = [
        CategoricalAccuracy(), Precision(), Recall()
    ]

    model = model_fn(num_classes=len(train_generator.class_indices), image_size=image_size)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

    callbacks = [
        EarlyStopping(min_delta=1e-5, patience=15, restore_best_weights=True),
        ModelCheckpoint(os.path.join(args.model_dir, 'best_model.h5'), save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001)
    ]

    model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks)
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
