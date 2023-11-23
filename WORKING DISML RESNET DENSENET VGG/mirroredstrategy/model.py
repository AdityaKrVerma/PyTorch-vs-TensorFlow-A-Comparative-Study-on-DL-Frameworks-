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

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

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

ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(32).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(tf.data.AUTOTUNE)

models = [DenseNet121, ResNet50, VGG16]
model_names = ['DenseNet121', 'ResNet50', 'VGG16']
epochs = 10

training_time = []
accuracy = []

strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

for idx, model_func in enumerate(models):
    model_name = model_names[idx]
    print(f"Training {model_name}")

    for iteration in range(2):
        print(f"Starting iteration {iteration + 1} of {model_name}")

        with strategy.scope():
            model = model_func(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            x = layers.Flatten()(model.output)
            x = layers.Dense(1024, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            output = layers.Dense(10, activation='softmax', dtype=tf.float32)(x)
            model = Model(model.inputs, output)

            initial_learning_rate = 0.001
            lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate,
                decay_steps=epochs * len(ds_train),
                end_learning_rate=0.0001,
                power=1.0,
                cycle=False
            )

            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

            model.compile(optimizer=optimizer,
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