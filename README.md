# Out-Of-Distribution Generalization on the Sine Wave Generator Task
This repository contains code and figures pertaining to an example of out-of-distribution (OOD) generalization on the sine wave generator task by a recurrent neural network (RNN). 

Introduced by [Sussillo and Barak (2013)](https://doi-org.libproxy.mit.edu/10.1162/NECO_a_00409), the sine wave generator task involves inputing a frequency value to an RNN and tasking the RNN to output a sine wave with the input frequency. In this repository, RNNs were trained to generate sine waves with frequencies ranging from $0.1$ to $0.6$ radians per unit time.

## Results
Occasionally, RNNs could generate sine waves with frequencies from $0.1$ to $0.6$ when only trained on a subset of those frequencies. 

RNNs were trained via a type of _curriculum learning_ in which frequencies were cumulatively added to the training set after a fixed number of training epochs had occured. The training set initially consisted of a frequency of $0.6$, and after every $25,000$ training epochs, a slower frequency was added. The training set expanded according to the following table:
| Model number | Training epoch | Frequencies in training set |
| --- | --- | ----------- |
| 1 | $[0,24999]$ | $0.6$ |
| 2 | $[25000,49999]$ | $0.6,0.5$ |
| 3 | $[50000,74999]$ | $0.6,0.5,0.4$ |
| 4 | $[75000,99999]$ | $0.6,0.5,0.4,0.3$ |
| 5 | $[100000,124999]$ | $0.6,0.5,0.4,0.3,0.2$ |
| 6 | $[125000,149999]$ | $0.6,0.5,0.4,0.3,0.2,0.1$ |

In the figure at the end of this document, we can see that model 4, when only trained on frequencies $0.6,0.5,0.4,0.3$, generalizes outside of its training distribution to generate reasonable sine waves for frequencies $0.2$ and $0.1$. Model 5 similarly generalizes to generate a sine wave for frequency $0.1$.

## Why can RNNs generalize on the generator task?
Before answering that question, notice how in the figure below that model 1 does not generate a reasonable sine wave for frequency $0.6$ despite undergoing $25,000$ training epochs. This is because the RNN only has $5$ recurrent neurons. For context, in the Sussillo and Barak paper where this task was introduced, their RNNs had $200$ recurrent neurons.

These results lead me to hypothesize that the [loss landscapes](https://openreview.net/forum?id=QC10RmRbZy9) of RNNs with few recurrent neurons are counter to our intuitions, even for simple tasks like the sine wave generator task. The loss landscapes for artificial neural networks (ANNs) with many neurons are thought to be smooth and contain many accessible, generalizable minima, though not necessarily OOD generalizable. In our example of 5 neuron RNNs optimized for the generator task, the loss landscape might be rough and contain relatively few minima that are diffcult for learning algorithms to reach; however, these minima appear to be OOD generalizable. Therefore, I don't believe that this OOD generalization phenomenon will be reproducible in ANNs of large numbers of neurons.

If anything, this OOD generalization phenomenon suggests that the number of neurons in an ANN could serve as a regularization parameter, with fewer neurons leading to a smaller, more general space of solutions. This phenomenon serves as a counterexample to the [double descent phenomenon](https://doi.org/10.48550/arXiv.2303.14151).

# 
<div align="center">
<img src="https://github.com/keith-murray/sine-wave-generator/blob/main/results/experiment_1/task_42/summary_plot.png" alt="OOD plot"></img>
</div>
