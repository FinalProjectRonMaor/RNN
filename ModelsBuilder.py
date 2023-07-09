import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, r2_score


# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     predictions = np.array([[round(num*6) for num in vals] for vals in predictions])
#     errors = np.abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / (test_labels + 1), axis=0)
#     accuracy = 100 - np.mean(mape)
#     print("predictions-", predictions,sep='\n')
#     print("test_labels-", test_labels,sep='\n')
#     return accuracy

def calculate_accuracyClassify6(y_test, y_pred):
    y_pred_rounded_and_grouped = np.round(y_pred)
    y_test_rounded_and_grouped = np.round(y_test)
    correct_predictions = np.sum(y_pred_rounded_and_grouped == y_test_rounded_and_grouped)
    total_predictions = y_test.shape[0]
    accuracy = correct_predictions / total_predictions * 100
    return accuracy

def calculate_accuracyClassify3(y_test, y_pred):
    y_pred_grouped = np.floor(y_pred / 1.66667)  # Grouping predictions
    y_test_grouped = np.floor(y_test / 1.66667)  # Grouping test values
    correct_predictions = np.sum(y_pred_grouped == y_test_grouped)  # Counting correct predictions
    total_predictions = y_test.shape[0]  # Total number of predictions
    accuracy = correct_predictions / total_predictions * 100  # Calculating accuracy
    return accuracy

emotions = {
    1: "Stress",
    2: "Joy",
    3: "Tired",
    4: "Sadness",
    5: "Fear",
    6: "Distraction",
    0: "Exit"
}
while True:
    choice = None
    while choice not in emotions:
        choice = int(input("Choose what type of emotion would you like to predict: \n 1. Stress \n 2. Joy \n 3. Tired \n 4. Sadness \n 5. Fear \n 6. Distraction \n 0. Exit \n"))
        if choice not in emotions:
            print("Invalid choice")

    if choice == 0:
        print("Bye :)")
        break
    choice = emotions[choice]
    print("You chose:", choice)

    # Load data
    dataset = pd.read_csv(f'Data{choice}.csv')
    # print(dataset.head())

    # Split data into input (X) and output (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1:].values

    # Scale data to a range of 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    # Split data into training and testing sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    X_train, X_test = X[0:train_size, :], X[train_size:len(dataset), :]
    y_train, y_test = y[0:train_size, :], y[train_size:len(dataset), :]

    # Reshape data to fit the LSTM input shape [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Rescale predictions and actual values back to original scale
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    # Evaluate model performance using mean squared error and R-squared score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    # Evaluate the model on the test data
    # accuracy = evaluate(model, X_test, y_test)
    # print('Accuracy:', accuracy)

    accuracy = calculate_accuracyClassify3(y_test, y_pred)
    print("Accuracy 3 Classes:", accuracy)

    accuracy = calculate_accuracyClassify6(y_test, y_pred)
    print("Accuracy 6 Classes:", accuracy)

