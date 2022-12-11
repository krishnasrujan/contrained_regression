import random

import numpy as np
import pandas as pd

from scipy import optimize



def BinaryCrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    term_0 = (1 - y_true) * np.log(1 - y_pred)
    term_1 = y_true * np.log(y_pred)
    return -np.mean(term_0 + term_1, axis=0)


class ConstrainedRegression():
    def __init__(self, lmbda=0, selected_features=[], log_error=True):
        self.selected_features = selected_features
        self.error_report = [] if log_error else None
        self.optimised_weights = []
        self.lmbda = lmbda
        self.res = None
        self.bounds = [(0, None) for feature in self.selected_features]  # bounds for minimise function

    @staticmethod
    def apply_activate_function(y_preds):
        y_preds = y_preds / 100
        return y_preds

    @staticmethod
    def get_simulated_weights(X, weights):
        if all([weight == 0 for weight in weights]):
            return np.zeros(X.shape)
        mask = np.invert(np.isnan(X))
        sim_weights = mask * weights
        sim_weights = sim_weights / np.sum(sim_weights, axis=1).reshape(-1, 1)
        return sim_weights

    @staticmethod
    def do_linear_combination(X, sim_weights):
        return np.sum(np.multiply(X, sim_weights), axis=1)

    def optimze(self, weights, *args):
        X = args[0]
        y = args[1]
        sim_weights = ConstrainedRegression.get_simulated_weights(X, weights)
        x_copy = X.copy()
        x_copy[np.isnan(x_copy)] = 0
        assert x_copy.shape == sim_weights.shape, print(x_copy.shape,
                                                        sim_weights.shape)
        y_preds = ConstrainedRegression.do_linear_combination(x_copy, sim_weights)
        y_preds = ConstrainedRegression.apply_activate_function(y_preds)
        cross_entropy = BinaryCrossEntropy(y, y_preds)
        self.error_report.append(cross_entropy)
        return cross_entropy + self.lmbda * np.sum(
            np.abs(np.nanmean(sim_weights, axis=0)))

    def fit(self, X_train, y_train, initial_weights=None):
        initial_weights = initial_weights if initial_weights else np.array(
            [0] * X_train.shape[1])
        self.res = optimize.minimize(
            self.optimze,
            x0=initial_weights,
            args=(X_train, y_train),
            bounds=self.bounds,
            method='L-BFGS-B',
            options={'maxfun': np.inf, 'maxls': 50, 'eps': 1e-15}
        )
        self.optimised_weights = self.res.x

    def predict(self, X_test, custom_weights=None):
        self.optimsed_weights = custom_weights if custom_weights else self.optimised_weights
        sim_weights = ConstrainedRegression.get_simulated_weights(X_test,
                                                             self.optimised_weights)
        x_copy = X_test.copy()
        x_copy[np.isnan(x_copy)] = 0
        assert x_copy.shape == sim_weights.shape, print(x_copy.shape,
                                                        sim_weights.shape)
        y_preds = ConstrainedRegression.do_linear_combination(x_copy, sim_weights)
        y_preds = ConstrainedRegression.apply_activate_function(y_preds)
        return y_preds

if __name__ == "__main__":
    df = pd.DataFrame({
        'variable_1': [50, 20, 10, np.nan, np.nan],
        'variable_2': [80, 60, 25, np.nan, 30],
        'target': [1, 1, 0, 0, 0]
    })
    selected_features = [col for col in df.columns if col != 'target']
    X_train = np.array(df[selected_features])
    y_train = np.array(df['target'])
    initial_weights = random.sample(range(0, 10), len(selected_features))
    constrained_reg_obj = ConstrainedRegression(selected_features=selected_features)
    constrained_reg_obj.fit(X_train, y_train, initial_weights)
    preds = constrained_reg_obj.predict(X_train)
    print(preds)