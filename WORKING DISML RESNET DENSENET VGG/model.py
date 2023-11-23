import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import VGG16, DenseNet121, ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time
from tensorflow.python.client import device_lib

assert tf.config.list_physical_devices('GPU'), "No GPU device found."

# Check available devices
print(device_lib.list_local_devices())

# Use the first available GPU
gpu_device = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu_device, True)

# Loading CIFAR-10 dataset
(ds_train, ds_test), ds_info = tfds.load(
    'cifar10',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(tf.cast(label, tf.int32), depth=10)  # one-hot encode the labels
    return image, label


train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rotation_range=20,
    fill_mode='nearest'
)

# Move models to GPU
with tf.device('/device:GPU:0'):
    ds_train = ds_train.map(preprocess)
    ds_train = ds_train.batch(16).prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.map(preprocess).batch(16).prefetch(tf.data.AUTOTUNE)

    models = [DenseNet121, ResNet50, VGG16]
    model_names = ['DenseNet121', 'ResNet50', 'VGG16']
    epochs = 10

    training_time = []
    accuracy = []

    for idx, model_func in enumerate(models):
        model_name = model_names[idx]
        print(f"Training {model_name}")

        for iteration in range(2):
            print(f"Starting iteration {iteration + 1} of {model_name}")

            model = model_func(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            x = layers.Flatten()(model.output)
            x = layers.Dense(1024, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            output = layers.Dense(10, activation='softmax')(x)
            model = Model(model.inputs, output)

            model.compile(optimizer=tf.keras.optimizers.legacy.SGD(lr=0.001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            start_time = time.time()
            model.fit(ds_train, epochs=epochs, verbose=2)
            elapsed_time = time.time() - start_time
            training_time.append(elapsed_time)

            test_loss, test_acc = model.evaluate(ds_test, verbose=2)
            accuracy.append(test_acc)

        del model
        tf.keras.backend.clear_session()

        mean_training_time = np.mean(training_time)
        mean_accuracy = np.mean(accuracy)
        std_training_time = np.std(training_time)
        std_accuracy = np.std(accuracy)

        print("Mean training time:", mean_training_time)
        print("Mean accuracy:", mean_accuracy)
        print("Standard deviation of training time:", std_training_time)
        print("Standard deviation of accuracy:", std_accuracy)