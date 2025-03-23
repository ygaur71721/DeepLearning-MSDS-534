import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint
import random
import io
from google.colab import drive

# Mount Google Drive for saving checkpoints
drive.mount('/content/drive')

# Create checkpoint directory
!mkdir -p /content/drive/MyDrive/text_generator_checkpoints

# Load data
path = "alice_in_wonderland.txt"
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()

# Clean text (optional)
import re
text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
text = re.sub(r'\s+', ' ', text)      # Replace multiple spaces with single space

# Create word-level vocabulary
words = text.split()
print('Total words:', len(words))
vocabulary = sorted(list(set(words)))
print('Unique words:', len(vocabulary))
word_indices = dict((w, i) for i, w in enumerate(vocabulary))
indices_word = dict((i, w) for i, w in enumerate(vocabulary))

# Cut the text into sequences of maxlen words
maxlen = 10
step = 1
word_sequences = []
next_words = []

for i in range(0, len(words) - maxlen, step):
    word_sequences.append(words[i: i + maxlen])
    next_words.append(words[i + maxlen])

print('Number of sequences:', len(word_sequences))

# Vectorize sequences
print('Vectorizing sequences...')
x = np.zeros((len(word_sequences), maxlen, len(vocabulary)), dtype=bool)
y = np.zeros((len(word_sequences), len(vocabulary)), dtype=bool)

for i, word_seq in enumerate(word_sequences):
    for t, word in enumerate(word_seq):
        x[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

# Define the LSTM model
print('Building model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(vocabulary))))
model.add(Dropout(0.2))
model.add(Dense(len(vocabulary), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Helper function to sample with temperature
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

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
'''
Commented the change reviewed by training the model on vocabulary changes

print('Training model...')
model.fit(x, y,
          batch_size=128,
          epochs=5,  # Reduced to 5 for faster execution
          callbacks=[print_callback, checkpoint])
'''
# Generate a longer text sample after training
print('Generating final sample...')
seed_text = ' '.join(words[:maxlen])
generated_text = generate_text(seed_text, num_generated=300, temperature=0.7)
print(generated_text)
