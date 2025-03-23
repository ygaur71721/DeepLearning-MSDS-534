import os
import numpy as np
import re
import shutil
import tensorflow as tf
import time

DATA_DIR = "./data"
CHECKPOINT_DIR = os.path.join(DATA_DIR, "checkpoints")
LOG_DIR = os.path.join(DATA_DIR, "logs")


def clean_logs():
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    shutil.rmtree(LOG_DIR, ignore_errors=True)


def download_and_read(urls):
    texts = []
    for i, url in enumerate(urls):
        p = tf.keras.utils.get_file("ex1-{:d}.txt".format(i), url, cache_dir=".")
        text = open(p, mode="r", encoding="utf-8").read()
        text = text.replace("\ufeff", "")
        text = re.sub(r'\s+', " ", text)
        texts.extend(text.strip().split())  # Tokenize by words
    return texts


def split_train_labels(sequence):
    input_seq = sequence[0:-1]
    output_seq = sequence[1:]
    return input_seq, output_seq


class CharGenModel(tf.keras.Model):

    def __init__(self, vocab_size, num_timesteps, embedding_dim, **kwargs):
        super(CharGenModel, self).__init__(**kwargs)
        self.embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.rnn_layer = tf.keras.layers.GRU(
            num_timesteps,
            recurrent_initializer="glorot_uniform",
            recurrent_activation="sigmoid",
            stateful=True,
            return_sequences=True
        )
        self.dense_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, x):
        x = self.embedding_layer(x)
        x = self.rnn_layer(x)
        x = self.dense_layer(x)
        return x


def loss(labels, predictions):
    return tf.losses.sparse_categorical_crossentropy(labels, predictions, from_logits=True)


def generate_text(model, prefix_string, word2idx, idx2word,
                  num_words_to_generate=100, temperature=1.0):
    input = [word2idx.get(w, 0) for w in prefix_string.strip().split()]
    input = tf.expand_dims(input, 0)
    generated = []
    model.reset_states()
    for i in range(num_words_to_generate):
        preds = model(input)
        preds = tf.squeeze(preds, 0) / temperature
        pred_id = tf.random.categorical(preds, num_samples=1)[-1, 0].numpy()
        generated.append(idx2word[pred_id])
        input = tf.expand_dims([pred_id], 0)
    return prefix_string + ' ' + ' '.join(generated)


# Run everything
texts = download_and_read([
    "http://www.gutenberg.org/cache/epub/28885/pg28885.txt",
    "https://www.gutenberg.org/files/12/12-0.txt"
])
clean_logs()

vocab = sorted(set(texts))
print("vocab size: {:d}".format(len(vocab)))

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
texts_as_ints = np.array([word2idx[w] for w in texts])
data = tf.data.Dataset.from_tensor_slices(texts_as_ints)

# Change these for Exercise 2.3
seq_length = 50  # Try 200 for large config
batch_size = 64  # Try 128 for large config

sequences = data.batch(seq_length + 1, drop_remainder=True)
sequences = sequences.map(split_train_labels)
dataset = sequences.shuffle(10000).batch(batch_size, drop_remainder=True)
steps_per_epoch = len(texts) // seq_length // batch_size

vocab_size = len(vocab)
embedding_dim = 256
model = CharGenModel(vocab_size, seq_length, embedding_dim)
model.build(input_shape=(batch_size, seq_length))
model.summary()

# TensorBoard logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=LOG_DIR,
    histogram_freq=0,
    write_graph=False,
    write_images=False,
    update_freq='epoch'
)

# Choose optimizer
model.compile(optimizer=tf.optimizers.Adam(), loss=loss)

num_epochs = 10  # Reduced for basic evaluation
for i in range(num_epochs // 5):
    start = time.time()
    model.fit(
        dataset.repeat(),
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        callbacks=[tensorboard_callback]
    )
    print("Training time for 5 epochs:", time.time() - start)

    checkpoint_file = os.path.join(CHECKPOINT_DIR, "model_epoch_{:d}".format((i + 1) * 5))
    model.save_weights(checkpoint_file)

    gen_model = CharGenModel(vocab_size, seq_length, embedding_dim)
    gen_model.load_weights(checkpoint_file)
    gen_model.build(input_shape=(1, seq_length))

    print("After epoch {:d}".format((i + 1) * 5))
    print(generate_text(gen_model, "Alice was beginning", word2idx, idx2word))
    print("---")
