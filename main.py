from tkinter import *
import random
from datetime import datetime
import MNISTConverter
import math

#adjustable constants
windowW = 600
windowH = 300
learningConstant = 0.00012
numDeepLayers = 1
#node number discluding the bias node
numDeepNodes = 815
#node number discluding the bias node
firstLayerNodeNum = 784
#total last layer nodes, no bias node in last layer
lastLayerNodeNum = 10

#creates window and canvas
titleA = "NNTest " + str(datetime.today())
titleB = titleA.replace(":", "-")
root = Tk()
root.title( titleB )
root.resizable(0 , 0)
canvas = Canvas(root, width=windowW, height=windowH)
canvas.pack()

# Network class will contain all methods and data for the neural network
class Network():
    lc = learningConstant
    def __init__(self):
        #list of last 100 attempts and if predicted answer on that iteration was corrct.
        self.hitOrMiss = []

        self.trainingNum = 0

        # activation vals are nested arrays
        # Think:
        # self.activationVals[Node's layer][Node's num in that layer]
        # --->activation val
        self.activationVals = []
        self.dActivationVals = []

        # weights are double-nested arrays.
        # Think:
        # self.weights[Node's layer][Node's num in that layer][Node num of next layer]
        # --->weight val
        self.weights = []
        self.dWeights = []

        self.assignActivationVals()
        self.assignWeights()

        # self.drawNodes()
        # self.drawWeights()

    # assigns zeros as place holders for the activationVals and dActivationVals lists
    # also assigns 1 for last node in each layer for bias nodes (except in the last layer, which doesn't have a bias node)
    def assignActivationVals(self):

        # first layer placeholders
        self.activationVals.append([])
        self.dActivationVals.append([])
        for i in range(firstLayerNodeNum):
            self.activationVals[0].append(0)
            self.dActivationVals[0].append(0)
        self.activationVals[0].append(1)

        #deep layer placeholders
        for i in range(1, numDeepLayers+1):
            self.activationVals.append([])
            self.dActivationVals.append([])
            for k in range(numDeepNodes):
                self.activationVals[i].append(0)
                self.dActivationVals[i].append(0)
            self.activationVals[i].append(1)

        #last layer placeholdlers
        self.activationVals.append([])
        self.dActivationVals.append([])
        for i in range(lastLayerNodeNum):
            self.activationVals[numDeepLayers + 1].append(0)
            self.dActivationVals[numDeepLayers + 1].append(0)


    # assigns a normal distribbution of random weights for the weights list and zeros as place holders for dWeights
    def assignWeights(self):
        # for every layer, except the last
        for i in range(len(self.activationVals)-1):
            self.weights.append([])
            self.dWeights.append([])
            #for every node
            for k in range(len(self.activationVals[i])):
                self.weights[i].append([])
                self.dWeights[i].append([])
                #for every node in he next layer
                for n in range(len(self.activationVals[i+1])):
                    isLastLayer = bool(i == len(self.activationVals)-2)
                    isLastNode = bool(n == len(self.activationVals[i+1])-1)
                    if (not isLastNode) or isLastLayer:
                        self.weights[i][k].append(random.gauss(0, 0.01))
                        self.dWeights[i][k].append(0)


    #applys the sigmoid activation function
    def sigActivation(self, x):
        if x < 0:
            return 1 - 1/(1 + math.exp(x))
        else:
            return 1/(1 + math.exp(-x))

    #applys the derivitive of the sig function
    def dSigActivation(self, x):
        y = self.sigActivation(x)
        return y*(1-y)

    def lReluActivation(self, x):
        return max(0.1*x, x)

    def dLReluActivation(self, x):
        if x > 0:
            return 1.0
        else:
            return .1



    # iteratively goes through each weight, multiplying the value of the weight and the sigmoid of the node value attatced, and adding it to the second node attatched
    def feedForward(self, input):
        for i in range(len(self.activationVals[0])-1):
            self.activationVals[0][i] = input[i]

        for i in range(len(self.weights)):
            for k in range(len(self.weights[i])):

                if i == 0:
                    signalStrength = self.activationVals[i][k]
                else:
                    signalStrength = self.lReluActivation(self.activationVals[i][k])

                #print(signalStrength)

                for n in range(len(self.weights[i][k])):
                    self.activationVals[i+1][n] += signalStrength * self.weights[i][k][n]

        print(str(self.activationVals[len(self.activationVals)-1]))

    #finds gradient of activation function for each node and weight
    def backProp(self, expectedVals):
        lastLayer = len(self.activationVals) - 1
        # dAcvtivaitonVals for last layer set
        for i in range(len(self.activationVals[lastLayer])):
            #the relative change in the loss function due to selected node
            dNodeError = 2*(self.sigActivation(self.activationVals[lastLayer][i]) - expectedVals[i])
            self.dActivationVals[lastLayer][i] = dNodeError

        # this segment is different b/c "k" represents nodes in the next layer, while "n" in the layer before.
        # this way, dAct and dSig only needed to bbe computed once for each node, instead of once for each weight
        reversedLayers = list(range(len(self.activationVals)))
        reversedLayers.reverse()
        reversedLayers.pop()
        #print(reversedLayers)
        for i in reversedLayers:
            #print("layer: ", str(i))
            for k in range(len(self.dActivationVals[i])):

                #get dAct and dSig for each node in selected layer
                # print(i, " ", k)
                dActFxn = self.dLReluActivation(self.activationVals[i][k])


                num = self.dActivationVals[i][k]*dActFxn


                #for each node in the layer before selected layer (in order to access weights connecting to selected layer's nodes)
                for n in range(len(self.activationVals[i-1])): # -1 b/c last node in layer is bias node, and bias node does not have weights leading into it
                    # print("at: ", i, " ", n, " ", k)
                    # print("dWeight limits: ", len(self.dWeights), " ", len(self.dWeights[i-1]), " ", len(self.dWeights[i-1][n]))
                    # print("actval limits: ", len(self.activationVals), " ", len(self.activationVals[i-1]))
                    # print()
                    if i == 1:
                        self.dWeights[i-1][n][k] = num * self.activationVals[i-1][n]
                    else:
                        self.dWeights[i-1][n][k] = num * self.lReluActivation(self.activationVals[i-1][n])
                    #print(self.dWeights[i-1][n][k])

                    if not n == len(self.activationVals[i-1])-1:
                        self.dActivationVals[i-1][n] += num * self.weights[i-1][n][k]



    #iterates through each weigth, adding a zero placeholder for each weight in the output of this function
    def initializeChanges(self):
        out = []

        for i in range(len(self.weights)):
            out.append([])
            for k in range(len(self.weights[i])):
                out[i].append([])
                for n in range(len(self.weights[i][k])):
                    out[i][k].append(0)

        return out

    #adds current dweights to argument passed through (the list of changes for thge batch)
    def adjustChanges(self, changes):
        for i in range(len(changes)):
            for k in range(len(changes[i])):
                for n in range(len(changes[i][k])):
                    changes[i][k][n] += self.dWeights[i][k][n]

    #writes relevant info into text file
    def writeStats(self, expectedOut):
        hOrM = 0
        lastLayer = self.activationVals[len(self.activationVals)-1]
        maxVal = self.activationVals[len(self.activationVals)-1][0]
        maxIndex = 0
        correctAnswer = 0
        for i in range(len(lastLayer)):
            if maxVal < lastLayer[i]:
                maxVal = lastLayer[i]
                maxIndex = i
            if expectedOut[i] == 1:
                correctAnswer = i

        if maxIndex == correctAnswer:
            hOrM = 1
        if self.trainingNum <= 99:
            self.hitOrMiss.append(hOrM)
        else:
            self.hitOrMiss[self.trainingNum%100] = hOrM
        hits=0
        misses=0
        for i in range(len(self.hitOrMiss)):
            if self.hitOrMiss[i] == 1:
                hits += 1
            else:
                misses += 1
        accuracy = hits/(misses+hits)


        loss = 0
        for i in range(len(lastLayer)):
            loss += (self.sigActivation(lastLayer[i])-expectedOut[i])**2


        path = "C:/815 Results/" + titleB + ".txt"
        f = open(path, "a")
        f.write(str(maxIndex) + ' ' + str(correctAnswer) + ' ' + str(hOrM) + ' ' + str(accuracy) + ' ' + str(loss) + '\n')
        print(str(maxIndex) + ' ' + str(correctAnswer) + ' ' + str(hOrM) + ' ' + str(accuracy) + ' ' + str(loss) + '\n')
        f.close()
        self.trainingNum += 1

    #sets all actVals dActVals and dWeights back to 0
    def cleanUp(self):
        for i in range(len(self.activationVals)):
            for k in range(len(self.activationVals[i])):
                self.activationVals[i][k] = 0
                if not k == len(self.activationVals[i])-1 or i == len(self.activationVals)-1:
                    self.dActivationVals[i][k] = 0
                else:
                    self.activationVals[i][k] = 1
                if not i == len(self.activationVals)-1:
                    for n in range(len(self.weights[i][k])):
                        self.dWeights[i][k][n] = 0


    #changes weights according to changes variable
    def changeWeights(self, changes, batchNum):
        for i in range(len(self.weights)):
            for k in range(len(self.weights[i])):
                for n in range(len(self.weights[i][k])):
                    self.weights[i][k][n] -= changes[i][k][n] * self.lc / batchNum

    #returns mnist image as list of ints
    def getInput(self, num):
        out = MNISTConverter.getImage("C:/Users/hatch/Desktop/atom workspace/character reading/train-images.idx3-ubyte", num)
        return out

    #returns list of expected output nodes
    def getOutput(self, num):
        out = [0] * 10
        ans = MNISTConverter.getLabel("C:/Users/hatch/Desktop/atom workspace/character reading/train-labels.idx1-ubyte", num)
        out[ans] = 1
        return out

    def batchCycle(self, batchNum):
        changes =  self.initializeChanges()
        for i in range(batchNum):
            rand = random.randint(0,8000)
            input = self.getInput(rand)
            expectedOut = self.getOutput(rand)
            self.feedForward(input)
            self.backProp(expectedOut)
            self.adjustChanges(changes)
            self.writeStats(expectedOut)
            self.cleanUp()
            print(str(self.trainingNum))
            #print()
            #self.drawInput(input)


        #self.printWeight(0,378,499)
        #self.printWeight(0,406,499)
        #self.printWeight(1,0,499)
        #self.printWeight(1,500,499)
        #self.printWeight(2,0,9)
        #self.printWeight(2,500,9)

        self.changeWeights(changes, batchNum)

    def drawNodes(self):
        for i in range(len(self.activationVals)):
            layer = ""
            for k in range(len(self.activationVals[i])):
                if (not i == len(self.activationVals)-1) and (k == len(self.activationVals[i])-1):
                    layer += 'o'
                else:
                    layer += 'x'
            print(layer)


    def drawDNodes(self):
        for i in range(len(self.dActivationVals)):
            layer = ""
            for k in range(len(self.dActivationVals[i])):
                layer += "x"
            print(layer)

    def drawWeights(self):
        print()
        print()
        for i in range(len(self.weights)):
            layer =""
            for k in range(len(self.weights[i])):
                for n in range(len(self.weights[i][k])):
                    layer+='x'
                layer += '  '
            print(layer)

    def drawInput(self, pic):
        for i in range(28):
            row = ""
            for k in range(28):
                if pic[i*28+k] == 0:
                    row = row + '.'
                else:
                    row = row + 'X'
            print(row)

    def printWeight(self, a, b, c):
        print("weight ", a, ',', b, ',', c, ':  ', self.weights[a][b][c])







brain = Network()

for i in range(2000):
    brain.batchCycle(10)
