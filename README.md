# Project 1
	Generally,  we find the K number of nearby points to the data we want to predict and take the majority of the data and consider that our data belongs to the majority of the data.

	Euclidean Distance -  Its the sum of the square root of the difference between the features of data with classes squared and the data which has only the features for which we want to predict the class(2 or 4) squared.
 	Now we list this sum with its class and sort it in ascending order so that we can take the top K 
	NOw is the crucial part ,WE don't want those feature's euclidean Distance after sorting them and taking the top K values ,Now all we want is the class of 
those top euclidean Distance ,We take those features by using i[1] for i in dist[:K]-this is taking the top K number of values in dist- which has the euclidean distance and its corresponding class .
	
	Thus i[1] will take the second part of the array that is for [25,2] it will take 2,which is the class and thats what we need.
Remember this is for K values that is K nearby neighbors so now we'll have an array of classes among which we should take the majority of the classes this is where Counter().mostcommon() comes in ,it'll take the classes that has most repeated and put them in an tuple within an array with the number of times it has repeated ,like this [(2,3),(4,2)]-this means 2 has repeated 3 times and 4 has repeated 2 times ,
so inorder to print the class we need the first of the of the array [0] and first of the tuple [0] so we type [0][0]


Now the accuracy part I really am confused with this is part ,I think what we do is add 1 point whenever the class we passed is one among the group which is 2 or 4 and divide this count by the total number of class 
present giving the percentage of accuracy ,similarly we can find the error .
 I have used Breast cancer data for predicting whether the cells are cancerous or malignant .
 K-Nearest Math is KNN from scratch 
 K-Nearest Breast cancer is KNN using scikit.
 The code is in python 3.