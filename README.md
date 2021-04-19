# Python implementation of simple neural networks for XOR gates

* Implemented without any external library including Numpy 
and other machine learning libraries.
* The neural networks comprise input, hidden, output layers.
* The only hidden layer has 2 hidden units with bias.
* The weights are updated by simple stochastic gradient descent 
algorithm which makes training unstable.
* Training might fail with about 10% chance due to randomly 
initialized weights and the poor ability of SGD. 
* SGD with MSE in PyTorch also failed to train this networks while 
Adam or other advanced optimizers made it.
* Training is sensitive to the learning rate. 
1.0 ~ 5.0 is recommended.