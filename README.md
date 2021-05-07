Gesture Classifier
==================
## Overview

This program recognizes the user's hand gesture and performs an assigned action on a Youtube video.

The actions currently recognized are:

Fist => To play a video

![fist](https://github.com/Tom2096/Gestures-YT-Scripts/blob/main/Imgs/fist.gif)

Open => To pause a video

![open](https://github.com/Tom2096/Gestures-YT-Scripts/blob/main/Imgs/open.gif)

Left => Go back to the previous video

![left](https://github.com/Tom2096/Gestures-YT-Scripts/blob/main/Imgs/left.gif)

Right => Skip to the next video

![right](https://github.com/Tom2096/Gestures-YT-Scripts/blob/main/Imgs/right.gif)

Check your webcam to ensure the models are loaded (This might take a while ... Wait until you see the black dots)

![isloaded](https://github.com/Tom2096/Gestures-YT-Scripts/blob/main/Imgs/isloaded.gif)


## Finished Product ##

[The finished product can be viewed here.](https://tom2096.github.io/Gestures-YT-React-App/) <--------------------------------------------------------------------------


**Important!**
Make sure that:
- You give permission for the website to use your camera
- You ensure that no other applications are currently using your camera

## Development

Firstly, the program takes input from the user's webcam and uses the [**Handpose Detection**](https://github.com/tensorflow/tfjs-models/tree/master/handpose) model provided by **TensorFlow.js** to discern 21 different hand landmarks.

The landmarks corresponding with different gestures (fist, open, left, right, ok) are then recorded, labeled, and stored to a json database.

The data is then collected by a custom dataset using **Python**, where the coordinates are normalized and trained using **Pytorch**. 

The model used to train the data consists of three linear layers that surround two activiation layers. It takes a tensor of shape (B by 42) as input and returns a tensor of shape (B by 5) as output (corresponding with the 5 different gestures).

The trained weights and the model are then converted into **ONNX** format to be used directly on React.

Below is the finished result:

## Pausing and Resuming ##

![fist and open](https://github.com/Tom2096/Gestures-YT-Scripts/blob/main/Imgs/pandr.gif)

## Skipping Videos ##

![left and right](https://github.com/Tom2096/Gestures-YT-Scripts/blob/main/Imgs/svideos.gif)


