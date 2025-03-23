import time

embedding_dim = 64
rnn_units = 128

# Two different settings for comparison
configs = [
    {"seq_length": 50, "batch_size": 32},
    {"seq_length": 100, "batch_size": 64}
]

for config in configs:
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    print(f"\nTraining with seq_length={seq_length}, batch_size={batch_size}")

    # Create dataset
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    def split_input_target(chunk):
        return chunk[:-1], chunk[1:]

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    dataset = dataset.take(100)  # Limit for faster training

    # Build and compile model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(vocab), embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(len(vocab))
    ])
    model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Time training
    start = time.time()
    model.fit(dataset, epochs=1, verbose=1)
    end = time.time()

    print(f"Training time: {round(end - start, 2)} seconds")
