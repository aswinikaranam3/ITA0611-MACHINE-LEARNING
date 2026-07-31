
e 4
e 4_

[2]
0s
import numpy as np

# 1. Activation function (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 2. Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

…    W1 += X.T.dot(d_hidden) * lr

    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

# 6. Testing
print("Final Output after Training:")
print(final_output)
Final Output after Training:
[[0.07304485]
 [0.93084242]
 [0.93122512]
 [0.07564609]]
