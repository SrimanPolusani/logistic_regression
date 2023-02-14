import numpy as np
import matplotlib.pyplot as plt
import math


class LogisticRegression:
    def __init__(self, X, y, w_init, b_init, alpha, num_iters):
        """
        :param X: An ndarray(m,n) with m examples and n features
        :param y: An ndarray(m) with m target values
        :param w_init: An ndarray(n) representing n initial values of model param w for each feature
        :param b_init: A scaler representing initial value of model param 'b'
        :param alpha: Learning rate
        :param num_iters: Total number of iterations in the gradient descent
        """
        self.X_train = X
        self.y_train = y
        self.w = w_init
        self.b = b_init
        self.learning_rate = alpha
        self.total_iters = num_iters

        # z and g(z) history and Cost [J(w, b)] history
        self.z_g_history = {}
        self.j_history = {}

        # To isolate all y = 1 examples and all y = 0 examples
        self.pos = self.y_train == 1
        self.neg = self.y_train == 0

        self.no_of_ex = self.X_train.shape[0]  # Number of examples
        self.no_of_features = self.X_train.shape[1]  # Number of features

    # <-----Cost Calculation----->
    def compute_cost_logistic(self):
        cost = 0
        for ex_index in range(self.no_of_ex):
            zi = np.dot(self.w, self.X_train[ex_index]) + self.b
            sigmoid = 1 / (1 + np.exp(-zi))  # 1/1+(e^-z)
            self.z_g_history[zi] = sigmoid
            cost_one = -self.y_train[ex_index] * np.log(sigmoid)
            cost_two = (1 - self.y_train[ex_index]) * np.log(1 - sigmoid)
            cost += cost_one - cost_two
        cost = cost / self.no_of_ex
        return cost

    # <-----dj_dw, dj_db Calculation----->
    def compute_gradient(self):
        dj_dw = np.zeros(self.no_of_features)
        dj_db = 0
        for ex_index in range(self.no_of_ex):
            zi = np.dot(self.w, self.X_train[ex_index]) + self.b
            sigmoid = 1 / (1 + np.exp(-zi))
            err = sigmoid - self.y_train[ex_index]
            for feature_num in range(self.no_of_features):
                dj_dw[feature_num] = dj_dw[feature_num] + (err * self.X_train[ex_index, feature_num])
            dj_db = dj_db + err
        dj_dw = dj_dw / self.no_of_ex
        dj_db = dj_db / self.no_of_ex
        return dj_dw, dj_db

    # <-----Performing Gradient Descent----->
    def gradient_descent(self):
        for iter_num in range(self.total_iters):
            w_gradient, b_gradient = self.compute_gradient()
            self.w = self.w - (self.learning_rate * w_gradient)
            self.b = self.b - (self.learning_rate * b_gradient)

            if iter_num < 100_000:  # To save resources
                self.j_history[self.compute_cost_logistic()] = [self.w, self.b]

            # To print w, b and cost 10 times in a gradient descent, irrespective of total number of iterations
            if iter_num % math.ceil(self.total_iters / 10) == 0:
                print('w: {}\nb: {}'.format(self.w, self.b))
                print('Iteration number: {}, Cost: {}\n'.format(iter_num, list(self.j_history.keys())[-1]))
        return self.w, self.b, self.j_history

    # <-----z Vs g(z) Sigmoid Graph----->
    def visualize_sigmoid(self):
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.scatter(self.z_g_history.keys(), self.z_g_history.values(), color='#1AA7EC')
        axes.set_title('Sigmoid Function', fontsize=15)
        axes.set_ylabel('Sigmoid g(z)', fontsize=15)
        axes.set_xlabel('Z', fontsize=15)
        plt.show()

    # <-----Xi Vs Y_train Graphs----->
    def visualize_xygraph(self):
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        for axis_num, axis in enumerate(axes):
            x_data_pos = self.X_train[self.pos][0:, [axis_num]]
            x_data_neg = self.X_train[self.neg][0:, [axis_num]]
            axis.scatter(x_data_pos, self.y_train[self.pos], marker='x', s=80, c='red', label='y=1', lw=3)
            axis.scatter(x_data_neg, self.y_train[self.neg], marker='o', s=100, facecolors='none', edgecolors='blue',
                         lw=3)
            axis.set_title(f'$X_{axis_num}$ Vs Y_train')
            axis.set_ylim(-0.08, 1.1)
            axis.set_xlabel(f'$X_{axis_num}$')
            axis.set_ylabel('Y_train')
        plt.show()

    # <-----Xo Vs X1 Graph and Decision Curve----->
    def visualize_result(self):
        fig, axes = plt.subplots(1, 1, figsize=(7, 6))
        axes.scatter(self.X_train[self.pos][0:, 0], self.X_train[self.pos][0:, 1], marker='x', s=100, c='red', lw=3)
        axes.scatter(self.X_train[self.neg][0:, 0], self.X_train[self.neg][0:, 1], marker='o', s=80, facecolors='none',
                     edgecolors='blue', lw=3)
        axes.plot([0, -self.b / self.w[0]], [-self.b / self.w[1], 0], color='#1AA7EC', lw=2)
        axes.set_title('$X_0$ ' + 'Vs ' + '$X_1$', fontsize=20)
        axes.axis([0, 3.5, 0, 3.0])
        axes.set_xlabel('$X_0$', fontsize=14)
        axes.set_ylabel('$X_1$', fontsize=14)
        plt.show()


X_tmp = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_tmp = np.array([0, 0, 0, 1, 1, 1])
w_tmp = np.zeros(X_tmp.shape[1])
b_tmp = 0.
alph = 0.1
iters = 10000

# Instance of the object
ml_object = LogisticRegression(X_tmp, y_tmp, w_tmp, b_tmp, alph, iters)
final_w, final_b, _ = ml_object.gradient_descent()
print('Final value of w: {}\nFinal value of b: {}'.format(final_w, final_b))
ml_object.visualize_sigmoid()
ml_object.visualize_result()
ml_object.visualize_xygraph()
