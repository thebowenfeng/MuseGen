from read_midi import Preprocess
from network import Bidirectional_Model

processed = Preprocess("alla-turca.mid")
mymodel = Bidirectional_Model(processed)
mymodel.train(70)
mymodel.generate(200, [1, 2, 3], "new.mid", 1000000)