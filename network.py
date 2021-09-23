import numpy as np
from read_midi import Preprocess
import tensorflow as tf
from mido import MidiFile, MidiTrack
from mido.messages.messages import Message
from mido import MetaMessage


SEQ_LEN = 10
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 100
RNN_UNITS = 1024
LSTM_UNITS = 150
TEMPERATURE = 1.0


class RNN_GRU(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)
        self.gru = tf.keras.layers.GRU(RNN_UNITS,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class RNN_Model:
    def __init__(self, preprocess: Preprocess):
        self.processed = preprocess
        self.dataset = None
        self.model = None

        self.create_dataset()

    def match_sequence(self, sequence):
        x = sequence[:-1]
        y = sequence[1: ]
        return x, y

    def create_dataset(self):
        id_tensor = tf.convert_to_tensor(self.processed.id_sequence)
        id_dataset = tf.data.Dataset.from_tensor_slices(id_tensor)
        sequences = id_dataset.batch(SEQ_LEN + 1, drop_remainder=True)
        self.dataset = sequences.map(self.match_sequence)
        self.dataset = self.dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(
        tf.data.experimental.AUTOTUNE)

    def train(self, epoch):
        self.model = RNN_GRU(self.processed.bag_size + 1)
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.model.compile(optimizer='adam', loss=loss)
        history = self.model.fit(self.dataset, epochs=epoch)

    def one_step_gen(self, inputs, states=None):
        predicted, states = self.model(inputs, states=states, return_state=True)
        predicted = predicted[:, -1, :]
        predicted = predicted / TEMPERATURE

        predicted_ids = tf.random.categorical(predicted, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        return predicted_ids.numpy()[0], states

    def generate(self, num_notes, prompt, output_file, tempo):
        next_id = np.array([prompt])
        states = None
        result = []

        for i in range(num_notes):
            next_id, states = self.one_step_gen(next_id, states=states)
            result.append(next_id)
            next_id = np.array([[next_id]])

        new_music = MidiFile()
        track = MidiTrack()
        new_music.tracks.append(track)

        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        for id in result:
            note_type, note, velocity, note_time = self.processed.id_bag[id].split(',')
            track.append(Message(note_type, note=int(note), velocity=int(velocity), time=int(note_time)))

        new_music.save(output_file)


class Bidirectional_Model:
    def __init__(self, processed: Preprocess):
        self.processed = processed
        self.dataset = None
        self.features = None
        self.labels = None
        self.targets = None
        self.model : tf.keras.models.Sequential = None

        self.create_dataset()

    def create_dataset(self):
        self.dataset = np.zeros(((len(self.processed.id_sequence) // SEQ_LEN) * (SEQ_LEN - 1), SEQ_LEN))

        counter = 0
        for i in range(0, len(self.processed.id_sequence), SEQ_LEN):
            if len(self.processed.id_sequence) - i < SEQ_LEN:
                break
            curr_sequence = np.array(self.processed.id_sequence[i:i+SEQ_LEN])
            for j in range(2, SEQ_LEN + 1):
                curr_ngram = curr_sequence[:j]
                padded = np.pad(curr_ngram, (SEQ_LEN - j, 0), mode='constant')
                self.dataset[counter] = padded
                counter += 1

        self.features, self.labels = self.dataset[:,:-1], self.dataset[:,-1]
        self.targets = tf.keras.utils.to_categorical(self.labels, num_classes=self.processed.bag_size + 1)

    def train(self, epoch):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Embedding(self.processed.bag_size + 1, EMBEDDING_DIM, input_length=SEQ_LEN - 1))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_UNITS)))
        self.model.add(tf.keras.layers.Dense(self.processed.bag_size + 1, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        history = self.model.fit(self.features, self.targets, epochs=epoch, batch_size=BATCH_SIZE)

    def generate(self, num_notes, prompt: list, output_file, tempo):
        result = prompt.copy()
        for i in range(num_notes):
            seq = np.array([result])
            seq = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=SEQ_LEN - 1, padding='pre')
            predicted = np.argmax(self.model.predict(seq, verbose=0), axis=-1)
            result.append(predicted[0])

        new_music = MidiFile()
        track = MidiTrack()
        new_music.tracks.append(track)

        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        for id in result:
            note_type, note, velocity, note_time = self.processed.id_bag[id].split(',')
            track.append(Message(note_type, note=int(note), velocity=int(velocity), time=int(note_time)))

        new_music.save(output_file)