import numpy as np

#Sigmoid 
def sigmoid(x, deriv=False):
    if (deriv == True):
        return 1 * (1 - x)
    return 1 / (1 + np.exp(-x))

sigmoid = np.vectorize(sigmoid)

#Input Layer
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1],
              [0,0,0],
              [1,1,0]])

#Output Layer
y = np.array([[0],
              [1],
              [1],
              [0],
              [1],
              [1]])

#Seeding the RNG
np.random.seed(1)

#Synapses (Hidden layers and weights)
syn0 = 2 * np.random.random((3,6)) - 1
syn1 = 2 * np.random.random((6,1)) - 1

#Training Steps
for j in range(60000):

    #Forward Propagation
    L0 = X
    L1 = sigmoid(np.dot(L0, syn0))
    L2 = sigmoid(np.dot(L1, syn1))

    #Calculate Loss
    L2_error = y - L2

    #Show loss between mean of Layer 2
    if(j % 10000) == 0:
        print ("Loss: " + str(np.mean(np.abs(L2_error))))

    #Back Propagation
    L2_delta = L2_error * sigmoid(L2, deriv=True)

    L1_error = L2_delta.dot(syn1.T)

    L1_delta = L1_error * sigmoid(L1, deriv=True)

    #Update Weights
    syn1 += L1.T.dot(L2_delta)
    syn0 += L0.T.dot(L1_delta)

print ("Output after training")
print (np.around(L2, 1))
