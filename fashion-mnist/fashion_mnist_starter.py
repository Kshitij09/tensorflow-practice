import tensorflow as tf
import tensorflow_datasets as tfds

tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
import math
from keras.backend.tensorflow_backend import set_session

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
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


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(train_examples / BATCH_SIZE))

test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(test_examples / 32))
print('Accuracy on test dataset:', test_accuracy)

for images, labels in train_dataset.take(1):
    print("Logits: ", model(images[0:1]).numpy())
