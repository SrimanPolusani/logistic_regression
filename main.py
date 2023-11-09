import numpy as np
import matplotlib.pyplot as plt
import math


class LogisticRegression:
    def __init__(self, txt_file, alpha, num_iters, lambda_):
        """
        :param txt_file: A txt file with training data
        :param alpha: alpha: Learning
        :param num_iters: Total number of iterations for the gradient descent
        :param lambda_: Regularization parameter
        """
        # Converts txt_file data into ndarray
        self.X_train = np.loadtxt(txt_file, usecols=(0, 1), delimiter=',')
        self.y_train = np.loadtxt(txt_file, usecols=2, delimiter=',')

        self.m_examples = self.X_train.shape[0]  # Number of examples
        self.n_features = self.X_train.shape[1]  # Number of features
        self.learning_rate = alpha
        self.total_iters = num_iters
        self.reg_param = lambda_

        # Zscore normalization, Formula: X_norm = X_train-mean/sigma
        self.X_train = (self.X_train - np.mean(self.X_train, axis=0)) / np.std(self.X_train, axis=0)

        # Generate initial params
        np.random.seed(1)
        self.w = np.random.randn(self.n_features)
        self.b = np.random.randn()

        # To isolate all y = 1 examples and all y = 0 examples
        self.pos = self.y_train == 1
        self.neg = self.y_train == 0

        # z and g(z) history and Cost [J(w, b)] history
        self.z_g_history = {}
        self.j_history = {}

    # <-----Cost Calculation----->
    def compute_cost_logistic(self):
        cost = 0
        for ex_index in range(self.m_examples):
            zi = np.dot(self.w, self.X_train[ex_index]) + self.b
            sigmoid = 1 / (1 + np.exp(-zi))  # 1/1+(e^-z)
            self.z_g_history[zi] = sigmoid
            cost_one = -self.y_train[ex_index] * np.log(sigmoid)
            cost_two = (1 - self.y_train[ex_index]) * np.log(1 - sigmoid + 1e-10)
            cost += cost_one - cost_two
        cost = cost / self.m_examples

        # Regularization Part
        reg_part = 0
        for j in range(self.n_features):
            reg_part += self.w[j] ** 2
        reg_part = reg_part * ((self.reg_param / self.m_examples) * 2)

        cost = cost + reg_part
        return cost

    # <-----dj_dw, dj_db Calculation----->
    def compute_gradient(self):
        dj_dw = np.zeros(self.n_features)
        dj_db = 0
        for ex_index in range(self.m_examples):
            zi = np.dot(self.w, self.X_train[ex_index]) + self.b
            sigmoid = 1 / (1 + np.exp(-zi))
            err = sigmoid - self.y_train[ex_index]
            for feature_num in range(self.n_features):
                dj_dw[feature_num] = dj_dw[feature_num] + (err * self.X_train[ex_index, feature_num])
            dj_db = dj_db + err
        dj_dw = dj_dw / self.m_examples
        dj_db = dj_db / self.m_examples

        # Regularization Part
        dj_dw += self.w * (self.reg_param / self.m_examples)
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
                print(
                    'Iteration number: {}, Cost: {}\n'.format(iter_num, list(self.j_history.keys())[-1])
                )
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
            axis.scatter(
                x_data_pos, self.y_train[self.pos],
                marker='x', s=80, c='red', label='y=1', lw=1
            )
            axis.scatter(
                x_data_neg, self.y_train[self.neg],
                marker='o', s=100, facecolors='none', edgecolors='blue',lw=1
            )
            axis.set_title(f'$X_{axis_num}$ Vs Y_train')
            axis.set_ylim(-0.08, 1.1)
            axis.set_xlabel(f'$X_{axis_num}$')
            axis.set_ylabel('Y_train')
        plt.show()

    def decision_curve(self):
        x1, y1 = 0, -self.b / self.w[0]
        x2, y2 = -self.b / self.w[1], 0
        a, b = y1 - y2, x2 - x1
        c = x1 * y2 - x2 * y1

        x1 = self.X_train[0:, 0].max()
        y1 = (-a * x1 - c) / b
        y2 = self.X_train[0:, 1].max()
        x2 = (-b * y2 + c) / a

        return (x1, y1), (x2, y2)

    # <-----Xo Vs X1 Graph and Decision Curve----->
    def visualize_result(self):
        p1, p2 = self.decision_curve()
        fig, axes = plt.subplots(1, 1, figsize=(7, 6))
        axes.scatter(
            self.X_train[self.pos][0:, 0], self.X_train[self.pos][0:, 1],
            marker='x', s=100, c='red', lw=3
        )
        axes.scatter(
            self.X_train[self.neg][0:, 0], self.X_train[self.neg][0:, 1],
            marker='o', s=80, facecolors='none', edgecolors='blue', lw=3
        )
        axes.plot([p1[0], p1[1]], [p2[0], p2[1]], color='#1AA7EC', lw=2)
        axes.set_title('$X_0$ ' + 'Vs ' + '$X_1$', fontsize=20)
        axes.set_xlabel('$X_0$', fontsize=14)
        axes.set_ylabel('$X_1$', fontsize=14)
        plt.show()


alph = 0.001
iters = 10_000
lambda_value = 0.1

# Instance of the object
ml_object = LogisticRegression('data1.txt', alph, iters, lambda_value)
final_w, final_b, _ = ml_object.gradient_descent()
print('Final value of w: {}\nFinal value of b: {}'.format(final_w, final_b))
ml_object.visualize_sigmoid()
ml_object.visualize_result()
ml_object.visualize_xygraph()
