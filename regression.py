import numpy as np
import pandas as pd
from scipy.stats import t

class regression:
    def __init__(self, intercept = True):
        self.X = X
        self.y = y
        self.keys = []
        self.beta = []
        self.variance = []
        self.r_squared = None
        self.intercept = 0
        self.has_intercept = intercept
        self.n, self.p = 0, 0
        self.p_values = []
        self.y_pred = []
        self.residuals = []
        self.y_mean = 0
        # SSR + SSE = SST
        self.SST, self.SSR, self.SSE = 0, 0, 0
        self.t_stat = []
    def fit(self, X: dict, y: dict):
        self.y_mean = np.mean(y, axis = 0)
        self.n = len(y) # fill attribute with number of observations
        ones = [1] * self.n
        ones = ('intercept', ones) # get data ready to put in 1 if intercept is true
        if self.has_intercept == True:
            items = list(X.items())
            items.insert(0, ones)
            X = dict(items)
        for key in X: # store keys in attribute
            self.keys.append(key)
        self.p = len(self.keys)
        self.y = np.array(y) # store data in attributes
        self.X = pd.DataFrame(X).values

        X = self.X
        y = self.y
        Xt = np.transpose(X)
        # preform (X'X)-1 X'y for betas
        XtX = np.matmul(Xt, X)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except:
            print('The data supplied creates a non-invertible matrix as it is a singular matrix with determinant zero')
            return
        Xty = np.matmul(Xt, y)
        betas = np.matmul(XtX_inv, Xty)
        for beta in betas:
            self.beta.append(beta)
        if self.has_intercept == True:
            self.intercept = self.beta[0]

        for idx in range(self.n):
            y_pred = np.matmul(self.beta, self.X[idx])
            resid = y[idx] - y_pred
            self.residuals.append(resid)
            self.y_pred.append(y_pred)
            self.SSR += (y_pred - self.y_mean) ** 2
            self.SSE += resid ** 2
        self.SST = self.SSE + self.SSR
        self.r_squared = 1 - (self.SSE / self.SST)

        sigma_square = self.SSE / (self.n - self.p)
        var_beta_hat = sigma_square * XtX_inv

        for var in range(len(var_beta_hat)):
            self.variance.append(var_beta_hat[var][var])
            self.t_stat.append(self.beta[var] / np.sqrt(self.variance[var]))
            self.p_values.append(2 * (1 - t.cdf(abs(self.t_stat[var]), self.n - self.p)))
            print(self.t_stat[var])

    def predict(self, X):
        temp = []
        for idx in range(len(X)):
            y_pred = np.dot(self.beta, X[idx])
            temp.append(y_pred)
        return temp

    def __str__(self):
        string = ''
        string += f'The coefficient of determination is {self.r_squared}\n'
        for idx in range(len(self.keys)):
            string += f'variable: {self.keys[idx]} has a beta estimate of {self.beta[idx]} with a variance of {self.variance[idx]} and p-value {self.p_values[idx]}\n'
        return string

data = {}
ct = 1
for outer in range(5):
    temp = []
    for idx in range(1, 51):
        temp.append((outer + (4 ** outer)) * idx * (np.random.normal(scale = 0.1) + 1))
    data[f'x{ct}'] = temp
    ct += 1
temp = []
for idx in range(1, 51):
    temp.append(10 * idx * (np.random.normal(scale = 0.1) + 1))
data['y'] = temp
print(data)
# data = {
#     'y': [100, 49, 57, 8, 58, 50, 65, 59, 78, 25, 73, 86, 99, 66, 14, 45, 11, 84, 70, 13, 100, 49, 41, 92, 88, 69, 34, 42, 53, 37, 11, 3, 91, 36, 79, 46, 24, 61, 48, 40, 62, 37, 91, 67, 71, 96, 76, 48, 42, 51, 98, 7, 98, 74, 67, 98, 12, 43, 70, 16, 24, 40, 80, 65, 38, 90, 25, 48, 78, 33, 45, 83, 18, 19, 18, 95, 77, 66, 70, 21, 17, 3, 36, 14, 18, 62, 21, 26, 80, 42, 21, 9, 79, 51, 68, 58, 29, 41, 6, 67],
#     'x1': [71, 67, 92, 40, 45, 40, 60, 92, 26, 42, 73, 41, 59, 75, 49, 88, 83, 25, 93, 29, 27, 71, 48, 6, 32, 45, 39, 96, 34, 2, 13, 4, 99, 92, 37, 36, 77, 55, 9, 92, 62, 14, 16, 19, 90, 27, 65, 48, 96, 54, 1, 14, 16, 65, 88, 7, 57, 68, 50, 70, 5, 95, 58, 6, 64, 29, 42, 24, 71, 65, 75, 14, 21, 42, 67, 56, 27, 19, 50, 57, 30, 64, 73, 23, 18, 87, 23, 54, 46, 97, 39, 51, 20, 85, 90, 65, 44, 67, 57, 48],
#     'x2': [88, 86, 83, 21, 74, 85, 92, 88, 58, 92, 22, 31, 95, 54, 35, 4, 70, 89, 87, 59, 92, 69, 83, 85, 94, 57, 96, 52, 15, 89, 18, 88, 16, 2, 71, 43, 21, 10, 20, 17, 83, 82, 74, 67, 18, 51, 68, 59, 8, 39, 73, 12, 89, 51, 61, 68, 87, 53, 92, 72, 50, 19, 65, 41, 39, 51, 46, 23, 98, 84, 10, 27, 47, 59, 19, 34, 57, 23, 32, 15, 79, 88, 18, 39, 67, 7, 78, 10, 38, 13, 66, 21, 11, 7, 24, 43, 8, 82, 88, 62],
#     'x3': [2, 91, 7, 98, 4, 42, 15, 21, 100, 53, 100, 10, 2, 57, 55, 8, 66, 96, 8, 100, 8, 11, 19, 82, 17, 62, 54, 68, 60, 37, 62, 76, 91, 87, 12, 22, 75, 76, 22, 19, 48, 94, 22, 44, 43, 15, 28, 1, 74, 58, 63, 39, 28, 30, 99, 44, 62, 95, 89, 16, 13, 85, 90, 87, 70, 90, 17, 9, 80, 44, 7, 54, 69, 100, 42, 81, 39, 49, 89, 84, 30, 66, 88, 12, 7, 34, 10, 97, 42, 36, 3, 26, 29, 4, 28, 60, 9, 53, 69, 50],
#     'x4': [13, 85, 26, 50, 83, 79, 69, 16, 60, 47, 26, 49, 37, 73, 24, 87, 29, 97, 99, 11, 10, 32, 90, 51, 40, 35, 31, 35, 82, 85, 35, 49, 75, 55, 11, 46, 22, 63, 67, 20, 56, 38, 56, 53, 60, 18, 95, 91, 70, 80, 37, 56, 58, 18, 32, 28, 16, 83, 89, 32, 25, 98, 57, 13, 52, 14, 96, 37, 72, 69, 36, 3, 39, 5, 14, 55, 95, 45, 47, 50, 60, 68, 97, 33, 88, 81, 61, 53, 24, 13, 79, 93, 16, 73, 8, 54, 38, 61, 21, 51]
# }

y = data.pop('y')
X = data
a = regression()
a.fit(X, y)
print(str(a))
# print(a.t_stat)
# print(a.beta)
print(a.predict([[1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12]]))
# for key in data:
#     print(key)
# print(len(X['x1']))
