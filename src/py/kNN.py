#	CS669 - Assignment 3 (Group-2) 
#	Last edit: 4/11/17
#	About: 
#		This program is a Bayes Classifier using k-nearest neighbour method using DTW distances between samples.

import numpy as np
import math
import os
import sys
			
dimension=2									#	Dimension of data vectors.

#	Calculates distance between two points in 'dimension' dimensional space.
def dist(x,y):
	distance=0
	for i in range(dimension):
		distance+=(x[i]-y[i])**2
	distance=math.sqrt(distance)
	return (distance)

#	Returns the DTW distance between two sequences of feature vectors.
def DTW(x,n,y,m):
	dtwArray=[[0.0 for i in range(m)] for j in range(n)]
	
	dtwArray[0][0]=dist(x[0],y[0])
	for i in range(n-1):
		dtwArray[i+1][0]=dtwArray[i][0]+dist(x[i+1],y[0])
	for i in range(m-1):
		dtwArray[0][i+1]=dtwArray[0][i]+dist(x[0],y[i+1])

	for i in range(n-1):
		for j in range(m-1):
			dtwArray[i+1][j+1]=dist(x[i+1],y[j+1])+min(dtwArray[i][j],dtwArray[i+1][j],dtwArray[i][j+1])

	return dtwArray[n-1][m-1]/(n*m)

#	Creates subdirectories if not present in a path.
def createPath(output):
	if not os.path.exists(os.path.dirname(output)):
		try:
			os.makedirs(os.path.dirname(output))
		except OSError as exc:
			if exc.errorno!=errorno.EEXIST:
				raise

#	Calculates the confusion matrix of all classes together.
def calcConfusion():
	confusionMatrix=[[0 for i in range(len(classes))] for i in range(len(classes))]
	for i in range(len(classes)):
		file=open(os.path.join(directoryO,"k"+str(k+1),classes[i]+"_values.txt"),"r")
		x=0
		for line in file:
			data=line.split()
			confusionMatrix[i][x]=int(data[1])
			x+=1
	return confusionMatrix

#	Returns first element as key for 'elem'.
def takeFirst(elem):
	return elem[0]

#	Program starts here...
print ("\nThis program is a Bayes Classifier using k-nearest neighbour method using DTW distances between samples.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for features training/test input and output or default (o/d): ")

direct=""
directO=""
directT=""
choiceIn='A'

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training feature data directory: ")
	directT=raw_input("Enter the path (relative or complete) of the test feature data directory: ")
	dimension=input("Enter the number of dimensions in the data (for input format, refer README): ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store results of the classification: ")
else:
	choiceIn=raw_input("Dataset (A/B): ")
	if choiceIn=='A' or choiceIn=='a':
		direct="../../data/Output/Dataset A/featureVectorsCH/train/"
		directT="../../data/Output/Dataset A/featureVectorsCH/test/"
		directO="../../data/Output/Dataset A/test_results/"
		dimension=24
	elif choiceIn=='B' or choiceIn=='b':
		direct="../../data/Output/Dataset B/featureVectorsSpeech/train/"
		directT="../../data/Output/Dataset B/featureVectorsSpeech/test/"
		directO="../../data/Output/Dataset B/test_results/"
		dimension=39
	else:
		print "Wrong input!. Exiting,"
		sys.exit()

if direct[len(direct)-1]!='/':
	direct+="/"
if directO[len(directO)-1]!='/':
	directO+="/"
if directT[len(directT)-1]!='/':
	directT+="/"

maxK=input("Enter the value of K upto which you want to test the data: ")

classes=[]

print "Calculating DTW distances. This will take a while..."
for contents in os.listdir(directT):
	contentName=os.path.join(directT,contents)
	if os.path.isdir(contentName):
		classes.append(contents)
		print "Inside test class - "+contents+"..."
		for filename in os.listdir(contentName):
			file=open(os.path.join(contentName,filename))
			testSequence=[]
			for line in file:
				number_strings=line.split()
				numbers=[int(num) for num in number_strings]
				testSequence.append(numbers)
			n=len(testSequence)
			print "Calculating DTW distances of all training samples from test sample - "+filename+"..."
			for contentsTrain in os.listdir(direct):
				createPath(os.path.join(directO,"distances_second_attempt",contents,filename,"total.txt"))
				outFileTotal=open(os.path.join(directO,"distances_second_attempt",contents,filename,"total.txt"),"a")
				contentTrainName=os.path.join(direct,contentsTrain)
				if os.path.isdir(contentTrainName) and contentsTrain!="use":
					for trainFilename in os.listdir(contentTrainName):
						trainFile=open(os.path.join(contentTrainName,trainFilename))
						trainSequence=[]
						for line in trainFile:
							number_strings=line.split()
							numbers=[int(num) for num in number_strings]
							trainSequence.append(numbers)
						m=len(trainSequence)
						DTWdistance=DTW(testSequence,n,trainSequence,m)
						outFileTotal.write(str(DTWdistance)+" "+trainFilename+" "+contentsTrain+"\n")
				outFileTotal.close()

directory=os.path.join(directO,"distances_second_attempt")
directoryO=os.path.join(directO,"results_second_attempt")
print "Sorting distances..."
for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName):
		for contentsTest in os.listdir(contentName):
			contentTestName=os.path.join(contentName,contentsTest)
			if os.path.isdir(contentTestName):
				file=open(os.path.join(contentTestName,"total.txt"),"r")
				outFile=open(os.path.join(contentTestName,"total_sorted.txt"),"w")
				Array=[]
				for line in file:
					number_strings=line.split()
					numbers=[]
					for i in range(len(number_strings)):
						if i==0:
							numbers.append(float(number_strings[i]))
						else:
							numbers.append(number_strings[i])
					Array.append(numbers)
				Array.sort(key=Key)
				for i in range(len(Array)):
					outFile.write(str(Array[i][0])+" "+Array[i][1]+" "+Array[i][2]+"\n")

#	Saving class names.
for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName):
		classes.append(contents)

print "Classifiying samples according to k-NN method..."
for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName):
		for contentsTest in os.listdir(contentName):
			contentTestName=os.path.join(contentName,contentsTest)
			if os.path.isdir(contentTestName):
				file=open(os.path.join(contentTestName,"total_sorted.txt"),"r")
				outFile=open(os.path.join(contentTestName,"classify_labels_for_k.txt"),"w")
				k=0
				freq=[0 for i in range(len(classes))]
				for line in file:
					number_strings=line.split()
					value=number_strings[len(number_strings)-1]
					for x in range(len(classes)):
						if value==classes[x]:
							freq[x]+=1
					maxFreq=0
					maxFreqInd=0
					for x in range(len(classes)):
						if freq[x]>maxFreq:
							maxFreq=freq[x]
							maxFreqInd=x
					outFile.write(classes[maxFreqInd]+"\n")
					k+=1
					if k==maxK:
						break
				file.close()
				outFile.close()

classesData=[]
for contents in os.listdir(directory):
	contentName=os.path.join(directory,contents)
	if os.path.isdir(contentName):
		classImages=[]
		for contentsTest in os.listdir(contentName):
			contentTestName=os.path.join(contentName,contentsTest)
			if os.path.isdir(contentTestName):
				imageData=[]
				file=open(os.path.join(contentTestName,"classify_labels_for_k.txt"),"r")
				for line in file:
					number_strings=line.split()
					imageData.append(number_strings[0])
				classImages.append(imageData)
				file.close()
		classesData.append(classImages)

print "Calculating results:"
for k in range(maxK):
	print "For k = "+str(k+1)+"..."
	for i in range(len(classesData)):
		createPath(os.path.join(directoryO,"k"+str(k+1),classes[i]+"_values.txt"))
		file=open(os.path.join(directoryO,"k"+str(k+1),classes[i]+"_values.txt"),"w")
		count=[]
		for j in range(len(classes)):
			x=[]
			x.append(classes[j])
			x.append(0)
			count.append(x)
			del x
		for x in range(len(classesData[i])):
			for y in range(len(classes)):
				if classesData[i][x][k]==count[y][0]:
					count[y][1]+=1
		for x in range(len(count)):
			file.write(count[x][0]+" "+str(count[x][1])+"\n")
		file.close()

	confusionMatrix=calcConfusion()
	Sumtot=0
	for i in range(len(classes)):
		for j in range(len(classes)):
			Sumtot+=confusionMatrix[i][j]

	confusionMatClass=[]
	for i in range(len(classes)):
		tempConfusionMatClass=[[0 for j in range(2)] for l in range(2)]
		sumin=0
		tempConfusionMatClass[0][0]=confusionMatrix[i][i]
		sumin+=tempConfusionMatClass[0][0]
		Sum=0
		for j in range(len(classes)):
			Sum+=confusionMatrix[i][j]
		tempConfusionMatClass[0][1]=Sum-tempConfusionMatClass[0][0]
		sumin+=tempConfusionMatClass[0][1]
		Sum=0
		for j in range(len(classes)):
			Sum+=confusionMatrix[j][i]
		tempConfusionMatClass[1][0]=Sum-tempConfusionMatClass[0][0]
		sumin+=tempConfusionMatClass[1][0]
		tempConfusionMatClass[1][1]=Sumtot-sumin
		confusionMatClass.append(tempConfusionMatClass)
	
	print "Data testing complete. Writing results in files for future reference..."
	filer=open(os.path.join(directoryO,"k"+str(k+1),"results.txt"),"w")
	
	print os.path.join(directoryO,"k"+str(k+1),"results.txt")

	filer.write("The Confusion Matrix of all classes together is: \n")
	for i in range(len(classes)):
		for j in range(len(classes)):
			filer.write(str(confusionMatrix[i][j])+" ")
		filer.write("\n")

	filer.write("\nThe Confusion Matrices for different classes are: \n")
	for i in range(len(confusionMatClass)):
		filer.write("\nClass "+str(i+1)+": \n")
		for x in range(2):
			for y in range(2):
				print str(confusionMatClass[i][x][y])
				filer.write(str(confusionMatClass[i][x][y])+" ")
			filer.write("\n")

	Accuracy=[]
	Precision=[]
	Recall=[]
	FMeasure=[]

	filer.write("\nDifferent quantitative values are listed below.\n")
	for i in range(len(classes)):
		tp=confusionMatClass[i][0][0]
		fp=confusionMatClass[i][0][1]
		fn=confusionMatClass[i][1][0]
		tn=confusionMatClass[i][1][1]
		accuracy=float(tp+tn)/(tp+tn+fp+fn)
		if tp+fp:
			precision=float(tp)/(tp+fp)
		else:
			precision=-1.0
		if tp+fn:
			recall=float(tp)/(tp+fn)
		else:
			recall=-1.0
		if precision+recall:
			fMeasure=2*precision*recall/(precision+recall)
		else:
			fMeasure=-1.0
		filer.write("\nClassification Accuracy for class "+str(i+1)+" is "+str(accuracy)+"\n")
		if precision!=-1.0:
			filer.write("Precision for class "+str(i+1)+" is "+str(precision)+"\n")
		else:
			filer.write("Precision for class "+str(i+1)+" is -\n")
		if recall!=-1.0:
			filer.write("Recall for class "+str(i+1)+" is "+str(recall)+"\n")
		else:
			filer.write("Recall for class "+str(i+1)+" is -\n")
		if fMeasure!=-1.0:
			filer.write("F-measure for class "+str(i+1)+" is "+str(fMeasure)+"\n")
		else:
			filer.write("F-measure for class "+str(i+1)+" is -\n")
		Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

	avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
	flagP,flagR,flagF=True,True,True
	for i in range (len(classes)):
		avgAccuracy+=Accuracy[i]
		if Precision[i]!=-1.0:
			avgPrecision+=Precision[i]
		else:
			flagP=False
		if Recall[i]!=-1.0:
			avgRecall+=Recall[i]
		else:
			flagR=False
		if FMeasure[i]!=-1.0:
			avgFMeasure+=FMeasure[i]
		else:
			flagF=False
	avgAccuracy/=len(classes)
	avgPrecision/=len(classes)
	avgRecall/=len(classes)
	avgFMeasure/=len(classes)

	filer.write("\nAverage classification Accuracy is "+str(avgAccuracy)+"\n")
	if flagP:
		filer.write("Average precision is "+str(avgPrecision)+"\n")
	else:
		filer.write("Average precision is -\n")
	if flagR:
		filer.write("Average recall is "+str(avgRecall)+"\n")
	else:
		filer.write("Average recall is -\n")
	if flagF:
		filer.write("Average F-measure is "+str(avgFMeasure)+"\n")
	else:
		filer.write("Average F-Measure is -\n")
	filer.write("\n**End of results**")
	filer.close()
	del confusionMatClass

#	End.