#KNN example


from numpy import *
import operator

# this creates dataset and labels
def createDataSet():
	group = array([[1.0, 1.1], [1.0,1.0], [0,0], [0,0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


# imported two modules
# 	first: NumPy: scientific computing package
#	second: operator modules: used in kNN algorithm for sorting


# Here: four pieces of data:
# 	Each: has two attirbutes or features, things we know about it.
#		In group matrix, each row is different piece of data.


def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet 
	sqDiffMat = diffMat**2					# this calculates distances
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):						# this votes with lowest k distnaces
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.iteritems(),
		key = operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]			#this sorts dictionary