import pandas as pd
from collections import Counter
import numpy as  np
import random

############################# Breast Cancer data scratch KNN ############################################

# definig the classifier for K nearest neighbor
def K_nearest(data ,predict ,k):
    if len(data) >= 3:
        print('Data is less than required to compare ! Idiot !')
    dist = []
    # eculidean distance
    for groups in data:
        for fet in data[groups]:
            euc_dist = np.sqrt(np.sum(np.array(fet)-np.array(predict))**2)
            # appending euc_distance into dist variable
            dist.append([euc_dist,groups])
    # sorted will sort the data in dist in ascending order based on the features euc distance and take the class alone [the crucial part in the conclusion]
    votes = [i[1] for i in sorted(dist)[:k]]

    #print(votes)
    # Counter will select the top most answer in votes
    vote_res = Counter(votes).most_common(1)[0][0]

    #print(vote_res)
    return vote_res



# a = [3,3,3,3,4,1,7,6,2,9]
#
# print(Counter(a).most_common(1)[0][0])




# read the data
data = pd.read_csv('breast-cancer-wisconsin.data.txt')
# removing id from the data
data = data.drop(['id'],1)
data.replace('?',-9999,inplace=True)
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



# populating the empty dictonary
for i in train_data:
    train_set[i[-1]].append(i[:-1])
# appending all the data except the last element in the data


for i in test_data:
    test_set[i[-1]].append(i[:-1])
    # appending the last data to test data ...because we dont want to test the same data that we train .


correct = 0
wrong = 0
total = 0


# passing the train data into the classifier
for group in test_set:
    #print(group)
    # passuing data in test_data one by one as training data to K_nearest neighbors
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


'''
# what do we do here ?
Generally,  we find the K number of nearby points to the data we want to predict and take the majority of the data and consider that our data belongs to the majority of the data.
# How do we do ?
Euclidean Distance -  Its the sum of the square root of the difference between the features of data with classes squared and the data which has only the features for which we want to predict the class(2 or 4) squared and 
 Now we list this sum with its class and sort it in ascending order so that we can take the top K 

NOw is the crucial part ,WE don't want those features euclidean Distance after sorting them and taking the top K values ,Now all we want is the class of 
those top euclidean Distance ,We take those features by using i[1] for i in dist[:K]-this is taking the top K number of values in dist- which has the euclidean distance and its corresponding class ,
Thus i[1] will take the second part of the array that is for [25,2] it will take 2,which is the class and thats what we need  
Remember this is for K values that is K nearby neighbors so now we'll have an array of classes among which we should take the majority of the classes this is where Counter().mostcommon() comes in ,it'll take the classes that has 
repeated and put them in an tuple within an array with the number of times it has repeated ,like this [(2,3),(4,2)]-this means 2 has repeated 3 times and 4 has repeated 2 times ,
so inorder to print the class we need the first of the of the array [0] and first of the tuple [0] so we type [0][0]


Now the accuracy part I really am confused with this is part ,I think what we do is add 1 point whenever the class we passed is one among the group which is 2 or 4 and divide this count by the total number of class 
present giving the percentage of accuracy ,similarly we can find the error .
 















'''



# ################################## Example_ to know what is KNN ###################################################
# d_points = { 'k':[[1,2],[2,3],[3,1]] ,'r':[[6,5],[7,7],[8,6]]}
#
# features = [5,10]
#
#
# # [[plt.scatter(ii[0],ii[1],s = 100 ,color='r')for ii in d_points[i]] for i in d_points]
# # plt.scatter(features[0],features[1])
# # plt.show()
#
#
# def K_nearest(data ,predict ,k=3):
#     if len(data) >= 3:
#         print('Data is less than required to compare ! Idiot !')
#     dist = []
#     for groups in data:
#         for fet in data[groups]:
#             euc_dist = np.sqrt(np.sum(np.array(fet)-np.array(predict))**2)
#             dist.append([euc_dist,groups])
#
#     votes = [i[1] for i in sorted(dist) [:k]]
#     print(Counter(votes).most_common(1))
#     vote_res = Counter(votes).most_common(1)[0][0]
#     print(vote_res)
# K_nearest(d_points,features)
#
