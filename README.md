Gesture Classifier
==================
## Overview

This program recognizes the user's hand gesture and performs an assigned action on a Youtube video.

The actions current recognized are:

Fist => To play a video

Open => To pause a video

Left => Go back to the previous video

Right => Skip to the next video

Currently there are only five Youtube videos in the playlist, but more can be easily added. 

[The finish product can be viewed here.](https://tom2096.github.io/Gestures-YT-React-App/) (**Note that you must give the app permission to use your camera**).

## Development

Firstly, the program takes input from the user's webcam and uses the **handpose** model proivided by **TensorFlow.js** to discern 21 different hand landmarks.

The landmarks corresponding with different gestures (fist, open, left, right, ok) are then recorded, labeled, and stored to a json database.

The data are then collected by a custom dataset using **Python**, where the coordinates are normalized and trained using **Pytorch**. 

Below is the model used to train the 21 differnet hand landmarks. It takes a tensor of shape (B by 42) as input and returns a tensor of shape (B by 5) as output (corresponding with the 5 different gestures).

The trained weights and the model are then compiled into **ONNX** format to be used directly on React.

Below is the finish result:
