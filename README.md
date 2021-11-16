# Simulate a pandemic with/without vaccination

We will simulate a model based on the 2009 H1N1 pandemic in Sweden, by performing a simulation of the 15 weeks from the 42nd of 2009 and the 5th of 2010 on a smaller scale model (scaled down by a factor of 104) retrieved from actual data.

# Implementation

🔗 More information about the experiments and discussion available in the [report](/report.pdf).

We will perform the pandemic starting from the same model developed in point 3, by
using as vaccination policy (derived from data of the H1N1 pandemic). In this scenario our goal will be to best estimate the infections behaviour of the real (scaled) pandemic, by performing simulations on a given set of parameters k; β; ρ and computing the root mean squared error (RMSE) between the average of the simulations.
At the start of each simulations, we will select at random 5% of the total population to be vaccinated, while 1 node will be chosen at random to be infected (chosen outside of the set of 5% vaccinated nodes to avoid a useless simulation). Since our goal is to best simulate the real pandemic, we will try to do so by searching for the parameters of k; β; ρ which yield the lowest RMSE. To find the best set of parameters, we try to simulate a gradient descent by exploring the neighborhood of each p

![example](/propagation.PNG)
<p align = "center">
Virus Propagation modeled with Random graph
</p>
