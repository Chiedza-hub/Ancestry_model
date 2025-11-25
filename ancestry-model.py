import pandas as pd
import numpy as np
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_ancestry_model(train, iterations, learning_rate):
    train_values = train.values
    num_features = train.shape[1] - 1  # Exclude label column
    weights = np.zeros(num_features + 1)  # Including bias term

    for i in range(iterations):
        grad = np.zeros(num_features + 1)
        for data_point in train_values: # eery datapoint without the index
            features = np.array(data_point)  # Convert to numpy array for easier manipulation     
            label = features[-1]  # Last element is the label
            # remove label from features and add bias term at the start
            features = np.insert(features[:-1], 0, 1)  # Add bias term
            print("features are:", features)
            z = sum(weights[i] * features[i] for i in range(len(features)))
            for j in range(len(features)):
                grad[j] += (label - sigmoid(z)) * features[j]       
        # update weights
        for j in range(len(weights)):
            weights[j] += learning_rate * grad[j]
    print("estimate weights are:", weights)
    return weights


def predict(test, weights):
    z = weights[0]  # bias term
    for i in range(len(test)):
        z += weights[i + 1] * test[i] 
    p_y_1 = sigmoid(z)
    predicted_label = 1 if p_y_1 > 0.5 else 0
    return predicted_label

def test_ancestry_model(test, weights):
    correct_predictions = 0
    total_predictions = test.shape[0] # .shape[0] gives number of rows

    for data_point in test.itertuples(index=False):
        features = np.array(data_point[:-1])  # All columns except the last one
        actual_label = data_point[-1]       # Last column is the label
        predicted_label = predict(features, weights)
        if predicted_label == actual_label:
            correct_predictions += 1
    print("model accuracy on tests:", correct_predictions / total_predictions)

def main():
    # Load the dataset
    df_tain = pd.read_csv('ancestry-train.csv')
    df_test = pd.read_csv('ancestry-test.csv')

    weights = fit_ancestry_model(df_tain, 1000, 0.0001) 
    test_ancestry_model(df_test, weights)

if __name__ == "__main__":
    main()