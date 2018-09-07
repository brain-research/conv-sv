# The Singular Values of Convolutional Layers  #

**Link to paper:[https://arxiv.org/pdf/1805.10408](https://arxiv.org/pdf/1805.10408 "The Singular Values of Convolutional Layers")**

Overview 
======
 
We characterize the singular values of the linear transformation associated with a convolution applied to a two-dimensional feature map with multiple channels. Our characterization enables efficient computation of the singular values of convolutional layers used in popular deep neural network architectures. It also leads to an algorithm for projecting a convolutional layer onto the set of layers obeying a bound on the operator norm of the layer.

Here, we provide the code for

* Our new method for calculating the singular values for 2D multi-channel conovlutional layers.	
* Time test comparing different methods for calculating the singular values.	
* Sketching the singular values for any trained model.	
* projecting a convolutional layer onto the set of layers obeying a bound on the operator norm of the layer.
	

### How do I get set up? ###

Requirements

+ Tensorflow
+ Numpy



## Citation ##

If you use this code, please cite our paper:


```
@article{
  sedghi2018singular,
  title={The Singular Values of Convolutional Layers},
  author={Sedghi, Hanie and Gupta, Vineet and Long, Philip M},
  journal={arXiv preprint arXiv:1805.10408},
  year={2018}
}
```

Authors 
======

* Hanie Sedghi ([hsedghi@google.com](hsedghi@google.com "mailto:hsedghi@google.com"))

* Vineet Gupta ([vineet@google.com](vineet@google.com "mailto:vineet@google.com"))

* Phil Long ([plong@google.com](plong@google.com "mailto:plong@google.com"))

This is not an officially supported Google Product.