import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random

# Importing data and replacing class string-types with int type.
df = pd.read_csv('iris.data')
df.replace('Iris-setosa', 1, inplace=True)  # replacing instances of "Iris-setosa" with the value of 1
df.replace('Iris-versicolor', 2, inplace=True)  # replacing all instances of "Iris-versicolor" with the value of 1
df.replace('Iris-virginica', 3, inplace=True)  # replacing all instances of "Iris-virginica" with the value of 1

accuracies = []

for test in range(1):

    def k_nearest_neighbours(data, predict, k=3):
        if len(data) >= k:
            warnings.warn("Error: The value of K must be greater than total voting groups.")
        distances = []
        for group in data:
            for features in data[group]:
                euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
                distances.append([euclidean_distance, group])

        votes = [i[1] for i in sorted(distances)[:k]]
        print(votes)
        vote_result = Counter(votes).most_common(1)[0][0]
        print(vote_result)
        confidence = Counter(votes).most_common(1)[0][1] / k
        print(confidence)

        return vote_result, confidence


    full_data = df.astype(float).values.tolist() # converted datatype of entire dataset to float, which is why we changed class values from str to int value above.
    random.shuffle(full_data)


    test_size = 0.2

    train_set = {1:[], 2:[], 3:[]}
    test_set = {1:[], 2:[], 3:[]}

    train_data = full_data[:-int(test_size*len(full_data))] # Making it so our training data portion will consist of the first 80% of the dataset, slicing the first 80% for use
    test_data = full_data[-int(test_size*len(full_data)):] # Slicing the last 20%

    for i in train_data:
        train_set[i[-1]].append(i[:-1]) # locates last column, being our label or what we want to predict and maps it to the train_set dictionary, accordingly - i.e. if a row or observation is 'iris-setosa' or '1' it will assign all the rows features to the '1' mapping.

    for i in test_data: # same as train_set except for test portion of data.
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print(f'Test: {test} - accuracy is {correct/total}')
    accuracies.append(correct/total)

print(f'\nThe average accuracy was: {sum(accuracies)/len(accuracies)}')
