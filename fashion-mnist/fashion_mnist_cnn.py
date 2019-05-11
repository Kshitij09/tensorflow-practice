import math
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm.auto
from keras.backend.tensorflow_backend import set_session
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard

tf.logging.set_verbosity(tf.logging.ERROR)

# Improve progress bar display
tqdm.tqdm = tqdm.auto.tqdm

# tf.enable_eager_execution()

# limit program eating up all the GPU memory

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_examples = metadata.splits['train'].num_examples
test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(train_examples))
print("Number of test examples:     {}".format(test_examples))


def normalize(images, labels):
    """
    The value of each image pixel will be an integer ranging from 0 to 255.
    For model to work properly and converge faster, we need to normalize these
    values to the range[0,1]
    """
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


class SimpleNet(keras.Model):

    def __init__(self, num_classes):
        super(SimpleNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation=tf.nn.relu,
                                         name="conv1")
        self.pool1 = keras.layers.MaxPool2D((2, 2), strides=2, name='pool1')
        self.conv2 = keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu, name='conv2')
        self.pool2 = keras.layers.MaxPooling2D((2, 2), strides=2, name='pool2')
        self.flatten = keras.layers.Flatten(name='flatten')
        self.dense1 = keras.layers.Dense(128, activation=tf.nn.relu, name='dense1')
        self.dense2 = keras.layers.Dense(10, activation=tf.nn.softmax, name='dense2')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)

        return output


model = SimpleNet(10)

BATCH_SIZE = 32
num_epochs = 5
train_dataset = train_dataset.repeat().shuffle(train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

optimizer = tf.train.AdamOptimizer(0.001)

model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model_name = 'ConvNet-maxpool3'
tensorboard = TensorBoard(log_dir="logs/{}".format(model_name), write_graph=True)

model.fit(train_dataset, epochs=5,
          steps_per_epoch=math.ceil(train_examples / BATCH_SIZE), callbacks=[tensorboard])

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(test_examples / 32))
print('Accuracy on test dataset:', test_accuracy)
