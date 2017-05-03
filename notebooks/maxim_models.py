import numpy as np
import scipy.optimize as opt

class many_weight_linear_model():
    def __init__(self, method='svm', delta=0.5, lmda=0.5):
        self.delta = delta
        self.lmda = lmda
        self.method = method
        self.W = None
        self.N, self.M, self.D = None, None, None
        self.v, self.m, self.t = None, None, None
        self.learning_rate, self.eps, self.beta_1, self.beta_2 = 1e-2, 1e-8, 0.9, 0.999
        
    def prepare_data(self, X, y):
        X = np.array(X)
        y = np.array(y).astype(int)
        X = np.concatenate((X, np.ones([X.shape[0], 1])), axis=1)
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.M = np.unique(y).shape[0]
        self.W = np.random.randn(X.shape[1], np.max(y) + 1) / np.sqrt(X.shape[1] / 2)
        self.t = 0
        self.m = np.zeros(self.W.shape)
        self.v = np.zeros(self.W.shape)
        return X, y

    def predict(self, X):
        X = np.concatenate((X, np.ones([X.shape[0],1])), axis=1)
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)
    
    def loss_and_grad_softmax(self, X, y):
        scores = X.dot(self.W)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        correct_logprobs = - np.log(probs[np.arange(X.shape[0]), y])
        loss = np.sum(correct_logprobs) + self.lmda * np.sum(self.W * self.W)
        
        dscores = probs
        dscores[np.arange(X.shape[0]), y] -= 1
        dscores /= X.shape[0]
        dW = np.dot(X.T, dscores) + self.lmda * 2 * self.W
        
        return loss, dW
    def loss_and_grad_svm(self, X, y):
        scores = X.dot(self.W)
        right_scores = scores[np.arange(self.N), y].reshape([self.N, 1])
        margins = np.maximum(0, scores - right_scores + self.delta)
        margins[np.arange(self.N), y] = 0
        loss = np.sum(margins) / self.N + self.lmda * np.sum(self.W * self.W)
        
        margins = margins > 0
        to_minus = np.sum(margins, axis=1)
        coefs_0 = ((margins[:, 0] > 0) - to_minus * (y == 0)).reshape([self.N, 1])
        derivatives = np.sum(X * coefs_0, axis=0)
        derivatives = derivatives.reshape([1, self.D])
        for i in np.arange(1, self.M):
            coefs_i = ((margins[:, i] > 0) - to_minus * (y == i)).reshape([self.N, 1])
            derivatives = np.concatenate((derivatives, [np.sum(X * coefs_i, axis=0)]))
        dW = derivatives.T / self.N + 2 * self.lmda * self.W
        
        return loss, dW
    
    def loss_and_grad(self, X, y):
        if self.method == 'softmax':
            return self.loss_and_grad_softmax(X, y)
        else:
            return self.loss_and_grad_svm(X, y)
    
    def adam(self, X, y):
        loss, dW = self.loss_and_grad(X, y)
        self.t += 1
        self.m = self.m * self.beta_1 + (1 - self.beta_1) * dW
        self.v = self.v * self.beta_2 + (1 - self.beta_2) * dW * dW
        
        m = self.m / (1 - self.beta_1 ** self.t)
        v = self.v / (1 - self.beta_2 ** self.t)
        self.W -= self.learning_rate * self.m / (np.sqrt(self.v) + self.eps)
    
    def fit(self, X, y):
        X, y = self.prepare_data(X, y)
        for step in range(1000):
            l = self.adam(X, y)

class one_weight_linear_model():
    def __init__(self, method='relax', eps=1, theta=1, steps_to_exclude=20, part_to_exclude=0.05):
        self.w = None
        self.method = method
        self.part_to_exclude = part_to_exclude
        self.steps_to_exclude = steps_to_exclude
        self.eps = eps
        self.theta = theta
        self.D = None
        self.N = None
        
    def prepare_data(self, X, y):
        X = np.array(X)
        self.N = X.shape[0]
        X = np.concatenate((X, np.ones([self.N, 1])), axis=1)
        self.D = X.shape[1]
        self.w = np.random.normal(0, 2. / self.D, self.D)
        X = np.apply_along_axis(lambda a: a * (-1) ** (y + 1), 0, X)
        return X
    
    def calc_relax_dw(self, X):
        negatives = X[X.dot(self.w) <= self.eps]
        scalar_mults = np.abs(negatives.dot(self.w))
        dw = np.sum(np.apply_along_axis(lambda a: a * scalar_mults, 0, negatives), axis=0)
        if np.any(np.abs(dw) > 1e-10):
            dw *= np.sum(scalar_mults ** 2) / np.sum(dw ** 2)
        else:
            dw *= np.sum(scalar_mults ** 2) / 1e-15
        return dw
    
    def exclude_most_often_unfulfilled(self, X, unfulfilled):
        for i in range(max(1, int(self.N * self.part_to_exclude))):
            num_to_del = np.argmax(unfulfilled)
            X = np.delete(X, num_to_del, 0)
            unfulfilled = np.delete(unfulfilled, num_to_del, 0)
        return X, unfulfilled
    
    def fit_using_relaxation(self, X, y):
        X = self.prepare_data(X, y)
        unfulfilled = np.zeros(self.N)
        step_number = 0
        while np.any(X.dot(self.w) < self.eps) and step_number < 500:
            dw = self.calc_relax_dw(X)
            self.w += self.theta * dw
            unfulfilled += X.dot(self.w) < self.eps
            step_number += 1
            if step_number % self.steps_to_exclude == 0:
                X, unfulfilled = self.exclude_most_often_unfulfilled(X, unfulfilled)
                
    def prepare_for_linprog(self, X, y):
        X = self.prepare_data(X, y)
        X = np.concatenate((- X, np.ones([self.N, 1])), axis=1)
        c = np.concatenate([np.zeros(self.D), [-1]])
        bounds = [[None, None] for i in np.arange(self.D + 1)]
        bounds[-1][1] = 1
        return X, c, bounds
                
    def exclude_with_largest_angles(self, X, res):
        unfulfilled = X[:, :-1].dot(res[:-1]) <= 0
        sumed = np.sum(X[unfulfilled, :-1], axis=0)
        cosines = np.ones(X.shape[0])
        for j in np.arange(len(unfulfilled)):
            if not unfulfilled[j]:
                continue
            rest_of_sum = sumed - X[j, :-1]
            cosines[j] = X[j, :-1].dot(rest_of_sum) / \
            rest_of_sum.dot(rest_of_sum) / X[j, :-1].dot(X[j, :-1])
        to_exclude = []
        for i in np.arange(max(1, self.part_to_exclude * self.N)):
            argmin = np.argmin(cosines)
            cosines[argmin] = 1
            to_exclude.append(argmin)
        X = np.delete(X, to_exclude, axis=0)
        return X
    
    def fit_using_linprog(self, X, y):
        X, c, bounds = self.prepare_for_linprog(X, y)
        res = np.nan
        while True:
            b_ub = np.zeros(X.shape[0])
            res = opt.linprog(c, A_ub=X, b_ub=b_ub, bounds=bounds).x
            if res[-1] > 0:
                break
            X = self.exclude_with_largest_angles(X, res)
        if res is not np.nan:
            self.w = res[:-1]
            
    def fit(self, X, y):
        if self.method == 'relax':
            self.fit_using_relaxation(X, y)
        elif self.method == 'linprog':
            self.fit_using_linprog(X, y)
        elif self.method == 'logistic':
            self.fit_logistic(X, y)
        else:
            print('No such method', self.method)
            
    def predict(self, X):
        X = np.array(X)
        return (X.dot(self.w[: -1]) + self.w[-1] >= 0).astype(np.int)

    def fit_logistic(self, X, y):
        X = self.prepare_data(X, y)
        self.w = opt.minimize(lambda w: self.calc_sigmoid_L(X, w), self.w, 
                             method='BFGS', jac=lambda w: self.calc_grad_L(X, w)).x
    def calc_sigmoid_L(self, X, w):
        return (1 / (1 + np.exp(X.dot(w)))).sum()
    def calc_grad_L(self, X, w):
        s = (1 / (1 + np.exp(X.dot(w))))
        return - X.T.dot(s * (1 - s))