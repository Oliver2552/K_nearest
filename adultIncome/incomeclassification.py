import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random

# importing data
df = pd.read_csv('adult.data')

# all column titles and values within columns have a leading space, so will remove.
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

# several columns have observations of '?' - replacing with 'unknown' or 'other' as there are many of them and removing would sacrifice a good chunk of data.
df['workclass'].replace('?', 'Unknown', inplace=True)
# df['occupation'].replace('?', 'Other', inplace=True)
# df['native_country'].replace('?', 'Unknown', inplace=True)

# dropping select columns
drop_col = ['workclass', 'fnlwgt', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_loss', 'native_country']
df.drop(drop_col, axis=1, inplace=True)

# Income is our predictor so will label-encode with 0 or 1
df['income'].replace('<=50K', 0, inplace=True)
df['income'].replace('>50K', 1, inplace=True)

# Normalising and one-hot-encoding numerical and categorical features, respectively.
# str_features = ['race','sex']
#
# str_encoded = pd.DataFrame()
# for feature in str_features:
#     encoded_feature = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
#     str_encoded = pd.concat([str_encoded, encoded_feature], axis=1)

int_features = ['age', 'education_num', 'capital_gain', 'hours_per_week']

int_normalised = df[int_features].copy()
for feature in int_features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    int_normalised[feature] = (df[feature] - min_val) / (max_val - min_val)

processed_df = pd.concat([int_normalised, df['income']], axis=1)


accuracies = []

# Implementing KNN
for i in range(2):
    def k_nearest_neighbours(data, predict, k=3):
        if len(data) >= k:
            warnings.warn("WARNING: the number of features exceeds the chosen value of k.")
        distances = []
        for group in data:
            for feature in data[group]:
                euclidean_distance = np.linalg.norm(np.array(feature) - np.array(predict))
                distances.append([euclidean_distance, group])

            votes = [i[1] for i in sorted(distances)[:k]]
            vote_result = Counter(votes).most_common(1)[0][0]
            confidence = Counter(votes).most_common(1)[0][1] / k

        return vote_result, confidence


    full_data = processed_df.astype(float).values.tolist()
    random.shuffle(full_data)


    # Performing a test/train split
    test_size = 0.2

    # Dictionaries to populate - Note: we're using dictionaries as our algorithm is designed to want a dictionary.
    train_set = {0:[], 1:[]} # dictionary mapping of lists associated with both classes; 0 (<=50K) and 1 (>50K) for training.
    test_set = {0:[], 1:[]} # Same as above but for test split.

    # Multiplying the 'test_size = 0.2' by the length of full data to create an index value and then we slice it based on that index value, which is converted to a whole number with 'int' - essentially, grabbing the first 80% of the rows.
    train_data = full_data[:-int(test_size*len(full_data))]
    test_data = full_data[-int(test_size*len(full_data)):] # Like above but slicing for the last 20% of data

    # Populating the dictionaries
    for i in train_data:
        train_set[i[-1]].append(i[:-1]) # searches for last element in each row and based on that appends it to the corresponding mapping in train_set
    for i in test_data:
        test_set[i[-1]].append(i[:-1]) # same as above but for test_set

    correct = 0
    total = 0

    # looping through each group/class in the test set, and then looping through each datapoint in each group, which is then fed into the KNN algorithm.
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbours(train_set, data, k=5)
            if group == vote:
                correct += 1
            total += 1
        print(f'The total accuracy is {correct/total}')
        accuracies.append(correct/total)

print(f'\n The averaged accuracy was: {sum(accuracies)/len(accuracies)}')




