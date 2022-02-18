# MuseGen

(Article on this project)[https://bowenfeng.tech/projects/ai/2021/09/29/musegen.html]

This project aims to train deep neural networks to generate music,
taking inspiration from techniques used in text generation.

## Demo

[A live, interactive website](http://18.224.212.77:8080/) of this project can be found here. Below is a video demo of the website:

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

## Future works

An interactive website to explore each model is being developed.
