from read_midi import Preprocess
from network import Bidirectional_Model, RNN_Model

processed = Preprocess("moonlight1.mid")
mymodel = Bidirectional_Model(processed)
mymodel.train(epoch=100)
mymodel.generate(num_notes=200, prompt=[1, 2, 3], output_file="new.mid", tempo=2000000)
# Tempo is a crochet (quarter note)'s length in microseconds.
