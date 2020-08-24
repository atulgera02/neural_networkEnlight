import numpy as np
#2009-2010 nba stats from https://www.kaggle.com/jacobbaruch/basketball-players-stats-per-season-49-leagues
# X = (total games played, total minutes played), y = field goals made total
# order: kevin durant, lebron james, kobe bryant, dwayne wade
x_all = np.array(([82, 3239.3], [76, 2965.6], [73, 2835.4], [77, 2792.4]), dtype=float)
y = np.array(([794], [768], [716]), dtype=float)
#x_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # input data
#y = np.array(([92], [86], [89]), dtype=float) # output
# scale units, Feature normalization
x_all = x_all/np.max(x_all, axis=0)
y = y/794 #most field goals made in the 2009-2010 nba season
#y = y/100
# split data
x_train = np.split(x_all, [3])[0] # training data
x_predicted = np.split(x_all, [3])[1]
#print(str(x_train))

#creating a neural network class
class neural_network(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        #weights
        self.Weights1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) weight matrix from input to hidden layer
        self.Weights2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer
    def sigmoid(self, s):
    # activation function
        return 1/(1+np.exp(-s))
    def forward(self, x_train):
    #forward propagation through our network
        self.z = np.dot(x_train, self.Weights1) # dot product of X (input) and first set of 2x3 weights
        self.z2 = self.sigmoid(self.z) # activation function

        self.z3 = np.dot(self.z2, self.Weights2)#dot of hidden and last set of weights
        o = self.sigmoid(self.z3) #activation function yielding the output
        return o
    def sigmoidPrime(self, s):  #derivative of sigmoid
        return s * (1 - s)
    def backward(self, x_train, y, o):
        self.o_error = y-o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) #apply derivative of sigmoid to the error
        self.z2error = self.o_delta.dot(self.Weights2.T) #z2 error: how much our hidden layer weights contributed to output error
        self.z2delta = self.z2error * self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error
        self.Weights1 += x_train.T.dot(self.z2delta) # adjusting first set (input --> hidden) weights
        self.Weights2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights
    def train(self, x_train, y):
        o = self.forward(x_train)
        self.backward(x_train, y, o)
    def predict(self):
        print("The predicted total # of field goals made is: ")
        print("input \n"+ str(x_predicted))
        print("output: \n" + str(self.forward(x_predicted) *764))
    def saveWeights(self):
        np.savetxt("w1.txt", self.Weights1, fmt="%s")
        np.savetxt("w2.txt", self.Weights2, fmt="%s")


nn = neural_network()
for i in range(20):
    print("Input: \n" + str(x_train))
    print("predicted output: " + str(nn.forward(x_train)))
    print("actual output: " + str(y))
    print("loss: \n" + str(np.mean(np.square(y-(nn.forward(x_train))))))
    print("\n")
    nn.train(x_train, y)

nn.saveWeights()
nn.predict()

