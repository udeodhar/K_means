"""
Title		: K means
Author		: Unmesh Deodhar
Description	: 
To run this code, you will have to modify the value of k as 10 or 30 and then it will do 5 iterations on its own
and it will save all the cluster center images of all iteration. Based on MSE of all iterations you should select
the minimum.
"""

import numpy as np
import pandas
import random
import math
from PIL import Image
import matplotlib.pyplot as plt

########Global variables containing all the digits and centers.
digits = []
testDigits = []
centers = []
k = 30

###############This class defines the centers
class Center:
        def __init__(self, array = None):
                self.dataPoints = []
                self.isChanged = False
                self.array = np.zeros((8,8))
                self.associatedClass = -1
                for i in range(8):
                        for j in range(8):
                             self.array[i,j] = random.randint(0,16)
	
###############This class defines training and testing digits
class Digit:
	def __init__(self, array = None):
                self.nearestCenter = -1
                self.array = np.zeros((8,8))
                for i in range(8):
                        for j in range(8):
                                self.array[i, j] = np.count_nonzero(array[(4*i):(4*i+4), (4*j) : (4*j+4) ])
                self.digitClass = array[32,0]

                        
##########Calculates euclidian distance between 2 arrays
def euclidianDistance(a, b):
        temp = a[:]-b[:]
        temp[:] = temp[:] * temp[:]
        sum1 = 0
        for i in range(8):
                for j in range(8):
                        sum1 += temp[i,j]
        return sum1
                
def main():
	print "Reading CSV"
	"""
	Pandas read csv
	"""
	#Reading training data
	inputData = np.array(pandas.read_csv("optdigits\optdigits\optdigits-pictures.train", header = None))
        testData = np.array(pandas.read_csv("optdigits\optdigits\optdigits-pictures.test", header = None))
        print testData.shape
	newInputData = np.zeros((126159,32), int)
        newTestData = np.zeros((59301,32), int)
        
	print "Data read!\nCleaning data..."
	
	for i in range (126159):
		a = 0
		for d in str(inputData[i,0]):
			if d != '\'' and d != ' ':
				newInputData[i, a] = d
				a = a + 1

	for i in range (59301):
		a = 0
		for d in str(testData[i,0]):
			if d != '\'' and d != ' ':
				newTestData[i, a] = d
				a = a + 1
	print "Data cleaned."
	
	#Restructuring the data to get the 2d array for each digit.....
	preCleanedData = np.reshape(newInputData, (3823,33,32))
	preCleanedTestData = np.reshape(newTestData, (1797,33,32))
		
	for digit in range(3823):
                digits.append(Digit(preCleanedData[digit,:,:]))

	for digit in range(1797):
                testDigits.append(Digit(preCleanedTestData[digit,:,:]))

	print "Data formatted."
		
        for iteration in range(5):
                print '------------------------------Iteration ',iteration,'----------------------------'
                centers = []
                for i in range(k):
                        centers.append(Center())

				flag = True
                print "Creating and startinng clusters..."
                while flag:
                        for i in range(k):
                                centers[i].dataPoints = []
                        #####Finding nearest clusters..
                        for i in range(3823):
                                minDistance = 16385
                                minIndex = -1
                                for j in range(k):
                                        dist = euclidianDistance(centers[j].array, digits[i].array)
                                        if minDistance > dist:
                                                minDistance = dist
                                                minIndex = j
                                digits[i].nearestCenter = minIndex
                                centers[minIndex].dataPoints.append(i)

						#####Calculating new clusters...
                        for i in range(k):
                                list1 = []
                                for j in centers[i].dataPoints:
                                        list1.append(digits[j].array)
                                temp = np.zeros((8,8))
                                for index1 in list1:
                                        temp += index1
                                if len(list1) != 0:
                                        tempArray = temp / len(list1)
                                else:
                                        tempArray = temp
                                if np.array_equal(centers[i].array, tempArray):
                                        centers[i].isChanged = False
                                else:
                                        centers[i].isChanged = True
                                centers[i].array = tempArray
                        flag = False
                        for i in range(k):
                                if centers[i].isChanged == True:
                                        flag = True

                print "Finished running clusters!\nYay!"

				########Calculating MSE,MSS,Entropy and Accuracy
				
				#Calculating MSE
                sumMSE = 0
                for i in range(k):
                        average = 0
                        for element in centers[i].dataPoints:
                                distance = euclidianDistance(digits[element].array, centers[i].array)
                                average += distance

                        if len(centers[i].dataPoints) != 0:
                                mse = average / len(centers[i].dataPoints)
                        else:
                                mse = 0
                        #print "MSE for cluster ",i,": ",mse

                        sumMSE += mse

                avgMSE = sumMSE / k
                print "Average MSE: ",avgMSE


                #Calculating MSS
                sumMSS = 0
                for i in range(9):
                        for j in range(i+1, k):
                                sumMSS += euclidianDistance(centers[i].array, centers[j].array)
                MSS = (sumMSS*2) / (k)*(k-1)
                print "MSS: ",MSS

                #Entropy
                classArray = np.zeros(10)
                entropy = 0
                for i in range(k):
                        for j in centers[i].dataPoints:
                                classArray[digits[j].digitClass] += 1
                        sumCluster = 0
                        for p in range(len(classArray)):
                                if len(centers[i].dataPoints) != 0:
                                        fraction = (classArray[p]/float(len(centers[i].dataPoints)))
                                else:
                                        fraction = 0
                                if fraction != 0:
                                        sumCluster += (fraction)*math.log(fraction,2)
                                else:
                                        sumCluster += 0
                        sumCluster = - sumCluster
                        entropy += sumCluster*len(centers[i].dataPoints)/3823.0
                print "Entropy of overall cluster: ",entropy
                
				
				#Accuracy
                frequencyofClasses = np.zeros(k)
                for i in range(k):
                        frequencyofClasses = np.zeros(k)
                        for element in centers[i].dataPoints:
                                frequencyofClasses[digits[element].digitClass] += 1
                        centers[i].associatedClass = np.argmax(frequencyofClasses)
                confusionMatrix = np.zeros((10,10))
                for i in range(1797):
                        minDistance = 16385
                        for j in range(k):
                                dist = euclidianDistance(centers[j].array, testDigits[i].array)
                                if minDistance > dist:
                                        minDistance = dist
                                        minIndex = j
                        confusionMatrix[testDigits[i].digitClass, centers[minIndex].associatedClass] += 1               

						
						
				############Visualizing clusters...........
                for i in range(k):
                        plt.gray()
                        plt.title(centers[i].associatedClass)
                        plt.imshow(centers[i].array)
                        plt.savefig("Iteration "+str(iteration)+"\\"+"Cluster "+str(i)+" Predicted class "+str(centers[i].associatedClass)+".png")

				
				################Printing all results
				print "Average MSE: ",avgMSE
                print "MSS: ",MSS
                print "Entropy of overall cluster: ",entropy
                print "Accuracy: ",(np.sum(np.diagonal(confusionMatrix))*100.0/np.sum(confusionMatrix))
                print confusionMatrix
				
				
main()
