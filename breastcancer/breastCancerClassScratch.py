import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random


accuracies = []

for i in range(2):
    def k_nearest_neighbours(data, predict, k=3):
        if len(data) >= k:
            warnings.warn('K is set to a value less than the total voting groups.')
        distances = []
        for group in data: # for either 'k' or 'r'
            for features in data[group]: # for either 'k' or 'r' mappings.
                euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) # Writing the explicit formula requires hardcoding the amount of classes to expect while this is faster and dynamic. Note: This method instead of math.sqrt
                distances.append([euclidean_distance, group])

        votes = [i[1] for i in sorted(distances)[:k]]
        vote_result = Counter(votes).most_common(1)[0][0] # 0th element of the 0th row - this measures accuracy
        confidence = Counter(votes).most_common(1)[0][1] / k

        return vote_result, confidence


    # Importing our data, replacing all 16 occurrences of missing data as denoted by '?', with '-9999' - such a large negative value is used to treat as an outlier. In addition, we drop the column 'id' as it holds no weight or value in label prediction (such large values will also cause
    # havoc on the model, more specifically, tank model accuracy.)
    df = pd.read_csv('breastCancerdata.data')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)
    full_data = df.astype(float).values.tolist() # Some values appear as string-type so we convert data to float-type. '.values' returns underlying data as a numpy array. '.tolist' converts the numpy array into a list of lists.
    random.shuffle(full_data)

    # Performing a train/test split
    test_size = 0.2

    # Dictionaries to populate - Note: we're using dictionaries as our algorithm is designed to want a dictionary.
    train_set = {2:[], 4:[]} # dictionary mapping of lists associated with both classes; 2 (benign) and 4 (malignant) for training.
    test_set = {2:[], 4:[]} # same as above but for testing

    train_data = full_data[:-int(test_size*len(full_data))] # Multiplying the 'test_size = 0.2' by the full data to create an index value and then we slice it based on that index value, which is converted to a whole number with 'int' - essentially, grabbing the first 80% of the rows.

    test_data = full_data[-int(test_size*len(full_data)):] # Much like above but slicing the last 20% of the data.

    # Populating train and test dictionaries:
    for i in train_data: # for every list/row in the train_data list
        train_set[i[-1]].append(i[:-1]) # 'i[-1] will search for the last element (or class in our case) of each row. train_set[i[-1]] uses the class label (last element of each row) as a key to access the according list in the dictionary mapping, train_set. '[:-1]' will then slice the last element.
        # As a whole: this appends the features of every row, not including the class label, to the appropriate list, 2 or 4, in train_set dictionary
    for i in test_data:
        test_set[i[-1]].append(i[:-1]) # same as above but for test data.

    correct = 0
    total = 0

    # looping through each group/class in the test set, and then looping through each datapoint in each group, which is then fed into the KNN algorithm. K is kept as the default value of 5.
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
    print(f'The total accuracy is {correct/total}')
    accuracies.append(correct/total)

print(f'\nThe averaged accuracy was: {sum(accuracies) / len(accuracies)}')
