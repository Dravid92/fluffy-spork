# Project 1 -Python Script for building KNN from scratch

## KNN - 
	Knn is an Supervised Machine learning that is used for creating classification models.
## How does it work ?
	Generally,  we find the distance between K number of nearby points to the data point we want to predict 
	Consider that the data point belongs to the majority of the K number nearest points 
### Example: 
- Consider an data set with 3 classes (c1,c2,c3)
- Let p be the data point we need to classify
- If we consider taking 5 nearby data points and find euclidean distance 
- Sort the data points in order of their distance 
- if 3/5 of the distances belong to a single cluster say c1 , we classify our data point(p) as the respective class(c1)
## What is Euclidean Distance ?
![Capture](https://user-images.githubusercontent.com/41041795/92879030-4e702e80-f42a-11ea-8ce8-164c767925d5.PNG)

### Euclidean Distance -  
- Its the square root of the sum of the squares of the difference between the features of data*(p)* and classes*(c1,c2,c3)*
- Now we list this sum with its class and sort it in ascending order 
- Classify *(p)* to the majority of the closest class among the k nearby points
## Explanation with Data :
### Data : [Wisconsin Breast Cancer Data set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29)
### Preprocessing :
Null values were imputed with -9999 so that when the data is fitted , the null values will be considered as [Outliers](https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm)


- Split the data into X(independent variables) and y(Classes) 
```python
X = np.array(data.drop(['Class'],1))
y = np.array(data['Class'])
```

- for cross validation , split the data into training and testing datasets.
```python
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.2)
```
-- test size parameter is the percentage of the total data to be taken as testing dataset
### Model : 
Creating a model using sklearn is pretty much simple 
It requries around 2 lines of python code.

```python
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)
```
- clf is the classifier and using the fit method we fit the model .
- Training datasets are passed in as parameters 

### Accuracy 
- score method inbuilt in the classifier is used to find the accuracy of the model
- test datasets are passed as parameters
- The model will predict the classes for the X_test and then compare it with the y_test 

$$ Accuracy = \frac{Number of correctly predicted classes}{Total Number of Observations that needed prediction}\ast $$

