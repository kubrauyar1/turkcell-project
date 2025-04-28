
#decide whether to go out or not
import numpy as np

#inputs
temperature = 5
humidity = 60

X = np.array([temperature,humidity])

#neuron #weights
weights = np.array([0.4,0.6])

#threshold value
bias = -20

#neuron output

output = np.dot(X,weights) + bias

print("The raw output of the neuron :", output)

def sigmoid(x):
    return 1/(1+np.exp(-x))

activated_output = sigmoid(output)

print("Neuron output after activation : ", activated_output)

