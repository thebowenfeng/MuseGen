import mido.messages.messages
from mido import MidiFile


class Preprocess:
    def __init__(self, midi_file):
        self.id_bag = {}
        self.bag_size = 0
        self.midi_messages = []
        self.hashed_messages = []
        self.id_sequence = []

        mid = MidiFile(midi_file, clip=True)
        for track in mid.tracks:
            for message in track:
                if type(message) == mido.messages.messages.Message and message.type in ["note_on", "note_off"]:
                    self.midi_messages.append(message)

        self.hash()
        self.process()

    def hash(self):
        self.hashed_messages = []
        for message in self.midi_messages:
            hash_string = f"{message.type},{message.note},{message.velocity},{message.time}"
            self.hashed_messages.append(hash_string)

    def find_from_id(self, hash_str):
        for key, val in self.id_bag.items():
            if val == hash_str:
                return key

        raise Exception("Hash string not in id bag")

    def process(self):
        self.id_sequence = []
        curr_id = 1

        for hash_str in self.hashed_messages:
            if hash_str not in self.id_bag.values():
                self.id_bag[curr_id] = hash_str
                self.bag_size += 1
                curr_id += 1

            self.id_sequence.append(self.find_from_id(hash_str))


#prep = Preprocess("moonlight1.mid")