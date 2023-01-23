import numpy as np


class LinearRegression:
    """
        A linear regression model.
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w : np.ndarray = None
        self.b : float = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the given data.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.

        """
        # add a column of 1s to X for bias
        X = np.column_stack((np.ones(len(X)), X))

        # calculate the coefficients
        xTx = np.dot(X.T, X)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X.T, y)
        self.w = np.dot(inverse_xTx, xTy)

        # set the intercept
        self.b = self.w[0]
        self.w = self.w[1:]

        return self
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        if (X.shape[1] != self.w.shape[0]):
            X = np.column_stack((np.ones(len(X)), X))

        y_pred = np.dot(X, self.w) + self.b
        return y_pred

class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression model that uses gradient descent to fit the model.
    """
    def __init__(self):
        self.w = None
        self.b = None



    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """
        Fit the model to the given data.

        Arguments:
            X (np.ndarray): The input data.
            y (np.ndarray): The output data.
            lr (float): The learning rate.
            epochs (int): The number of epochs to train for.

        """

         # add a column of 1s to X for bias
        X = np.column_stack((np.ones(len(X)), X))

        # initialize coefficients and intercept
        n_features = X.shape[1]
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

        # perform gradient descent
        for i in range(epochs):
            print("in interation", i)
            y_pred = self.predict(X)
            errors = y - y_pred
            self.w += lr * np.dot(X.transpose(), errors) / len(X)
            self.b += lr * errors.sum() / len(X)
        print("ended for loop")
        

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
       # X = np.column_stack((np.ones(len(X)), X))

        # make predictions
        
        if (X.shape[1] != self.w.shape[0]):
            print("X shape is", X.shape)
            print("w shape is", self.w.shape)
            X = np.column_stack((np.ones(len(X)), X))

        print(X.shape)
        print(self.w.shape)
        y_pred = np.dot(X, self.w) + self.b
        print(y_pred.shape)
        return y_pred
