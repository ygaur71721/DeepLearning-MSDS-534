
# Function to generate text
def generate_text(seed_text, num_generated=100, temperature=0.5):
    print('Generating text with seed:', seed_text)

    # Get the last maxlen words from seed text
    seed_words = seed_text.lower().split()[-maxlen:]
    if len(seed_words) < maxlen:
        # Pad with the beginning of the text if seed is too short
        seed_words = (words[:maxlen-len(seed_words)] + seed_words)[-maxlen:]

    generated = ' '.join(seed_words)

    for i in range(num_generated):
        x_pred = np.zeros((1, maxlen, len(vocabulary)))
        for t, word in enumerate(seed_words):
            if word in word_indices:
                x_pred[0, t, word_indices[word]] = 1

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_word = indices_word[next_index]

        generated += ' ' + next_word
        seed_words = seed_words[1:] + [next_word]

    return generated

# Print a sample after each epoch
def on_epoch_end(epoch, logs):
    # Function to generate text samples after each epoch
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    seed_idx = random.randint(0, len(words) - maxlen - 1)
    seed_text = ' '.join(words[seed_idx: seed_idx + maxlen])

    for temperature in [0.2, 0.5, 1.0]:
        print('----- Temperature:', temperature)
        print(generate_text(seed_text, num_generated=50, temperature=temperature))
        print()

# Define callbacks
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
checkpoint_dir = "/content/drive/MyDrive/text_generator_checkpoints"
# Change the filepath to end with .keras
filepath = f"{checkpoint_dir}/word_level_model-epoch-loss.keras"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


# Train the model with a smaller number of epochs for Colab
print('Training model...')
model.fit(x, y,
          batch_size=128,
          epochs=5,  # Reduced to 5 for faster execution
          callbacks=[print_callback, checkpoint])

# Generate a longer text sample after training
print('Generating final sample...')
seed_text = ' '.join(words[:maxlen])
generated_text = generate_text(seed_text, num_generated=300, temperature=0.7)
print(generated_text)



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
