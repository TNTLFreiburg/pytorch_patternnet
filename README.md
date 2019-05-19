# pytorch_patternnet
In this repository you can find a pytorch implementation of PatternNet as described in [Learning how to explain neural networks: PatternNet and PatternAttribution](https://arxiv.org/abs/1705.05598). The idea of PatternNet is to visualize what a neural network sees as the signal in an input. For this a backward projection from the output space to the input space of the network is created from example forward projections. 

## Installation 
In order to run the code provided in this repository it is easiest to set up a conda environment from the *patternnet_env.yml* file. If you have conda installed, simply download this repository, navigate to the folder where you saved the repository in your terminal and execute the following :

``` 
conda env create -f environment.yml
```

## How to use
At the moment a PatternNet can be created via the PatternNet class implemented in networks.py. For this initialize your PatternNet with a list of your networks\' layers in the order that they are used during the forward pass of your network (`patternnet = PatternNet(net.layers`)). Layer types that can be used are __Conv2d__, __Linear__, __ReLU__ and __MaxPool2d__. 

After initialization of your PatternNet you have to compute the patterns used for the backward projection. This can be done by executing the `patternnet.compute_statistics(data)` command, where _data_ is some input data of the task that the network was trained on. After computing the statistics the patterns have to be computed and set. Just execute `patternnet.compute_patterns()` and `patternnet.set_patterns()` to do this. 

When the patterns are set the signal of some input can be computed by just supplying the patternnet with the input: `signal = patternnet(input)`. 

An example for a PatternNet creation and usage can be found in *PatternNet_Mnist_Example.ipynb*.
