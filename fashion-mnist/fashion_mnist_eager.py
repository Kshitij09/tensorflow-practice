import tensorflow as tf
import tensorflow_datasets as tfds

tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe
from keras.backend.tensorflow_backend import set_session

# limit program eating up all the GPU memory
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
model.call = tfe.defun(model.call)


def loss(model, inputs, labels):
    logits = model(inputs)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
        return loss_value, tape.gradient(loss_value, model.trainable_weights)


def accuracy(logits, labels):
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(
        tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size


def log_summaries(loss_value, accuracy_value):
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("loss", loss_value)
        tf.contrib.summary.scalar("accuracy", accuracy_value)


BATCH_SIZE = 32
num_epochs = 5
train_dataset = train_dataset.shuffle(train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

optimizer = tf.train.AdamOptimizer()

# keep track for plotting
loss_history = []
accuracy_history = []

global_step = tf.train.get_or_create_global_step()
logdir = 'logs'
model_name = 'mnist-dense-128'
summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10000, name=model_name)
summary_writer.set_as_default()

for epoch in range(num_epochs):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()
    for (batch, (images, labels)) in enumerate(train_dataset):
        loss_value, grads = grad(model, images, labels)
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step=global_step)

        # track progress
        epoch_loss_avg(loss_value)
        epoch_accuracy(tf.argmax(model(images), axis=1, output_type=tf.int64), labels)

    # end of epoch
    loss_history.append(epoch_loss_avg.result())
    accuracy_history.append(epoch_accuracy.result())
    print("Epoch {:03d}: Loss: {:.4f} Accuracy: {:.3%}".format(epoch + 1,
                                                               epoch_loss_avg.result(),
                                                               epoch_accuracy.result()))
    # logging it to tensorboard
    log_summaries(epoch_loss_avg.result(),
                  epoch_accuracy.result())

# Visualizing the results
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(loss_history)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(accuracy_history)
plt.show()
