import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

accuracies = []

for i in range(25):
    # Importing our data, replacing all 16 occurrences of missing data as denoted by '?', with '-9999' - such a large negative value is used to treat as an outlier. In addition, we drop the column 'id' as it holds no weight or value in label prediction (such large values will also cause
    # havoc on the model, more specifically, tank the model accuracy.)
    df = pd.read_csv('breastCancerdata.data')
    df.replace('?', -9999, inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    # Creating X and Y values for features and label - Y is the label, which is the class - what we're trying to classify.
    X = np.array(df.drop(['class'], axis=1))
    y = np.array(df['class'])

    # splitting dataset into training and testing portions where test size will be 0.2 or 20% of the data.
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Initiallising the classifer and setting it to KNN and specifying the 80% X and y training set
    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    # Scoring our model - Note: you can also measure confidence however, we're measuring accuracy.
    accuracy = clf.score(X_test, y_test)
    print(f'The model achieved an accuracy of {accuracy}%')

    # While you could pickle the classifier, since it trains so fast there is no point. Let's make a prediction on made-up data:
    example_data = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,2,2,4,2,2]])
    example_data = example_data.reshape(len(example_data), -1) # Note: we will need reshape our entry - we can adapt to varying sizes of inputs by taking the input of the example_data
    prediction = clf.predict(example_data)
    print(prediction)
    accuracies.append(accuracy)

print(f'\nThe averaged accuracy was: {sum(accuracies) / len(accuracies)}')