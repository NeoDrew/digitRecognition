# digitRecognition

Neural network for handwritten digit detection

## How to use

Simpily download the python code in `app.py` & run. 

_Note, torch is only compatible with python versions 3.7._

## Dependencies

```bash
pip3.7 install torch torchvision matplotlib
```

## Description

The model trains on the famous [MNIST handwritten digits dataset](https://en.wikipedia.org/wiki/MNIST_database). The dataset contains 60,000 28x28 grayscale images with their coresponding class values [0-9]. 

Neural networks are computational models inspired by the human brain's architecture, designed to recognize patterns and make decisions. 

Composed of interconnected nodes organized into layers, this network consist of an input layer, hidden layers, and an output layer. During training, the network learns by adjusting the weights assigned to connections between nodes to minimize the difference between predicted and actual outputs. This process involves forward propagation, where input data passes through the network to produce a prediction, and backward propagation, which adjusts the weights based on the error. Backpropagation calculates the gradient of the error with respect to the network's weights and updates them using optimization algorithms like gradient descent. Through iterations of training with labeled data, the neural network optimizes its parameters to enhance its ability to generalize and accurately classify new, unseen inputs.
