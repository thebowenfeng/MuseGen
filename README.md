# MuseGen

This project aims to train deep neural networks to generate music,
taking inspiration from techniques used in text generation.

## Demo

[A live, interactive website](http://3.138.86.8:8080/) of this project can be found here. Below is a video demo of the website:

[![Video](http://img.youtube.com/vi/-TX8kUK7zos/0.jpg)](http://www.youtube.com/watch?v=-TX8kUK7zos)

[Source code of the website](https://github.com/thebowenfeng/MuseGenWebsite/) can be found here.

## Usage

#### Requirements

All libraries should be updated to the newest version at the time,
to avoid compatibility and deprecation issues.

- Tensorflow 2
- Mido

See main.py for sample usage. This project contains two RNN models,
GRU and Bidirectional LSTM. Their usage is exactly the same. Hyperparameters
are found in network.py and can be tuned.

## Rationale

Music and speech is similar in many ways. Both are considered "sequence data",
meaning both almost always appear in sequences, and each data point is dependent
on the previous datapoint. Both also contain "sequential patterns", 
where there will exists patterns of small sequences of data. 

RNNs, or Recurrent Neural Networks, are especially suited to train
sequential data, such as time-series data (e.g avg rainfall in a
given year) or texts, as discussed above. By having the capability
to store "states", RNNs can effectively "remember" previous 
inputs, which makes the predicted output meaningful when taken
as a whole, rather than just a random string of data.

In this project, two models were developed and tested. GRU (gated
recurrent unit) and Bidirectional LSTM (long short term memory).
Details of both models are discussed below.

## GRU

GRU is a really popular variant of RNNs, alongside with LSTM, is
frequently used for text-based tasks such as semantic analysis or
text generation. Compared to its counterpart, GRU is newer 
and is slightly faster to train, which is why I opted for GRU
as opposed to vanilla LSTM. 

The entire music is first serialized into a sequence of integers,
each integer represents a specific note. The master sequence is then
divided into smaller subsequence, which simulates a "verse" of 
music. Smaller sequence length leads to no continuity and erratic/random
notes being generated, whereas larger subsequence will lead to
higher repetition of the original music. As such, different music
will have different optimal subsequence length, requiring some
trial and error.

Each subsequence is then mapped to its neighboring note, which is inspired
by a similar technique used in text generation. Essentially, every note is mapped
to the next note that appears. Intuitively, this allows the neural network
to recognize inherent patterns between the progression of different notes.
For example, if there are a lot of arpeggios in a particular piece of music,
then the sequence mapping will reflect this by having more "pairs"
that form thirds. Even if we decided to randomly pick a pair out of all
pairs, then it is likely that we will pick a pair that forms a third.

The problem with this technique, is that unlike textual data, music is often highly
repetitive. For instance, it is extremely unlikely for a given word/sentence
to have a high frequency of a certain repeated pattern of characters. However,
such is not the same case with music. In music, artifacts such as
trills will result in what I refer to as an "infinite loop of hell" where the 
network will generate a infinitely long trill because there are a lot of pairs
of alternating notes, due to there being several trills. The neural network does
not possess an understanding for music, so therefore it does not know when to 
stop "trilling". You can observe this happening, sometimes, when the 
neural network is trained on Beethoven's "For Elise". The famous
E & D# "trill" at the beginning will cause the infinite loop of hell to happen,
where the AI will generate a infinitely long E & D# "trill".

The structure of the network is fairly standard: Embedding layer, GRU layer
and a Dense output layer. The generated music is not perfect by a long shot,
but will retain notable features from the music. For instance, when trained on 
Beehoven's Moonlight Sonata 1st Movement, the famous G# C# E trio 
will likely to frequently repeat itself. However, when trained on more
erratic music such as Listz's La Campanella, the output will resemble to something
closer to a keyboard being mashed together.

## Bidirectional LSTM

Bidirectional LSTM is a newer and more promising variant of a Unidirectional
(normal) LSTM. In essence, it is a combination of two RNNs. One that goes 
forward, and one that goes backwards. From a higher level perspective, this
will allow your network to retain both states from the future and the past at
any given point, making it understand context better.

For this model, we will utilize a different data processing technique.
The music is still serialized into a sequence of integers, and split
into smaller subsequence. However, each subsequence is converted into
a set of n-grams, which again is inspired by a common technique used to
process textual data.

For instance, given the sequence of notes A, B, C, D, E. The n-gram
sequence is:
- A, B
- A, B, C
- A, B, C, D
- A, B, C, D, E

In this case, the last element in each n-gram will become the "target"
value.

This will give the neural network opportunity to learn from both a note's
immediate neighbor (see GRU model) and also the larger "mother" sequence
it was apart of. In other words, the neural network should, in theory,
preserve both continuity of the notes, and avoid duplication of the original
music, to a certain degree. (see below)

However, this means that there will be more emphasis on continuity when compared
to the GRU model, as each n-gram set will only have 1/2 "neighboring notes"
mappings, whereas the rest will be a longer sequence. Although this is unlikely
to result in strong plagiarism, it is expected that the output
music will retain more snippets of the original music, and less
"creativity". 

All n-grams are left-padded and the last notes are stripped to
be used as target values. The model is similar to the GRU model,
Embedding and Dense layers are the same, the only difference
is the GRU layers is replaced with a Bidirectional layer.

As somewhat expected, the output is slightly more structured,
but more similar to the original music, compared to GRU model. However,
the difference is not major, and more erratic music (La Campanella) will
still result in gibberish output.

## Final thoughts

Although similar, there are still key differences between textual data
and musical data. Given the simplicity of both the training data, and
the model, the output was surprisingly good. It is fairly obvious
that neither model truly grasped the technicalities of music.

## Future works

An interactive website to explore each model is being developed.
