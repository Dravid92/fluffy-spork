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

- Its the square root of the sum of the squares of the difference between the features of data *(p)* and classes *(c1,c2,c3)*
- Now we list this sum with its class and sort it in ascending order 
- Classify *(p)* to the majority of the closest class among the k nearby points

## KNN Using SKlearn :

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
- test size parameter is the percentage of the total data to be taken as testing dataset
### Model : 

Creating a model using sklearn is pretty much simple .It requries around 2 lines of python code.

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

		Accuracy =( Number of correctly predicted classes )  * 100
		           --------------------------------------
		      Total Number of Observations that needed prediction

- Accuracy was around **97 %**

# Math Behind KNN (Python) 

### Preprocessing :

```python
# read the data
data = pd.read_csv('breast-cancer-wisconsin.data.txt')

# removing id from the data
data = data.drop(['id'],1)

# replace null values by -9999
data.replace('?',-9999,inplace=True)
```
- The only difference here is that we split the train and test dataset manually and using **dictonary** to store them

```python
# passing data as float and changing them to list
df = data.astype(float).values.tolist()

# shuffling the data
random.shuffle(df)

# amount of data for test and training
test_size = 0.2

# empty dictonary and populate it later
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

# slicing the data
    # the data except the last 20 % of the data
train_data = df[:-int(test_size*len(df))]

    # the last 20 % of the data as test data
test_data = df[-int(test_size*len(df)):]
``` 
- Once  data needed for the training and the testing is determined we populate the dictonary with the observations based on their class (2 and 4) 
- The data is split based on classes for calculating the accuracy 

```python 
# populating the empty dictonary
for i in train_data:
    train_set[i[-1]].append(i[:-1])
# appending all the data except the last element in the data which is the class


for i in test_data:
    test_set[i[-1]].append(i[:-1])
    # appending the class(y) to test data dictonary  
```
### Model 

- The input for this KNN math model function are the X variable **data** , the Classes that needed to be predicted **predict** and  the number distances to be considered for classification **k**

```python
# definig the classifier for K nearest neighbor
def K_nearest(data ,predict ,k):

    if len(data) >= 3:
        print('Data is less than required to compare ! Idiot !')
    dist = []
    
    ## eculidean distance ##
    for groups in data:
        for fet in data[groups]:
            euc_dist = np.sqrt(np.sum(np.array(fet)-np.array(predict))**2)
    ##                   ##
    
            # appending euc_distance into dist variable
            dist.append([euc_dist,groups])
	    
    # sorted will sort the data in dist in ascending order based on the features euc distance and take the class alone [the crucial part in the conclusion]
    votes = [i[1] for i in sorted(dist)[:k]]

    #print(votes)
    # Counter will select the top most answer in votes
    vote_res = Counter(votes).most_common(1)[0][0]

    #print(vote_res)
    return vote_res
``` 
### Accuracy 

- For every *correct,wrong prediction* the **correct, wrong variable** below will be updated and the **total variable** shows the *total number of observations that are considered for prediction*
```math 
 Accuracy = correct / total
 Error = Wrong / total 
```

```Python
correct = 0
wrong = 0
total = 0


# passing the train data into the classifier
for group in test_set:
    #print(group)
    # passing data in test_data one by one as training data to K_nearest neighbors
    
    for features_2 in test_set[group]:
    
        #print(features_2)
        #print(train_set)
        # prints out the nearest data point
        vote = K_nearest(train_set,features_2,5)

        if group == vote:
            correct += 1
        else:
            wrong += 1
        total += 1
	
print('Acc : ', correct/total)
print('err :',wrong/total)
