import tensorflow as tf
import tensorflow_datasets as tfds

tf.logging.set_verbosity(tf.logging.ERROR)
# Improve progress bar display
import tqdm.auto

tqdm.tqdm = tqdm.auto.tqdm
import matplotlib.pyplot as plt
import math
from tensorflow.python import keras
from keras.layers import *
# from keras.layers import *
from keras.models import Model
from keras.backend.tensorflow_backend import set_session

# limit program eating up all the GPU memory
tf.enable_eager_execution()
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


def preview_image():
    for image, label in test_dataset.take(1):
        break
    image = image.numpy().reshape((28, 28))

    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()


class LeNet(Model):

    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.num_classes = num_classes
        self.conv1 = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1))
        self.maxpool = MaxPooling2D((2, 2), strides=2)
        self.conv2 = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(400, activation='relu')
        self.dense2 = Dense(120, activation='relu')
        self.dense3 = Dense(84, activation='relu')
        self.dense4 = Dense(self.num_classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)

        return x


model = keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary()

BATCH_SIZE = 32
num_epochs = 5
train_dataset = train_dataset.repeat().shuffle(train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

optimizer = tf.train.AdamOptimizer(0.001)
# model = LeNet(num_classes=10)
model.compile(optimizer='adam',
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# keep track for plotting
# loss_history = []
# accuracy_history = []
#
# global_step = tf.train.get_or_create_global_step()
# model_name = 'LeNet-maxpool'
# tensorboard = TensorBoard(log_dir='./logs/{}'.format(model_name),
#                           write_graph=True)

model.fit(train_dataset, epochs=5,
          steps_per_epoch=math.ceil(train_examples / BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(test_examples / 32))
print('Accuracy on test dataset:', test_accuracy)
