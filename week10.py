import numpy as np
import random


# 활성화 함수 - 1. 시그모이드
def sigmoid(x, derivative=False):
    if derivative == True:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


# 활성화 함수 - 2. tanh
def tanh(x, derivative=False):
    if derivative == True:
        return 1 - x**2
    return np.tanh(x)


# 가중치 배열을 만드는 함수
def makeMatrix(i, j, fill=0.0):
    mat = []
    for i in range(i):
        mat.append([fill] * j)
    return mat


data = [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]

# 실행 횟수(iterations), 학습률(lr), 모멘텀 계수(mo) 설정
iterations = 5000
lr = 0.01
mo = 0.9


class NeuralNetwork:
    def __init__(self, num_x, num_yh, num_yo, bias=1):
        # 입력 값(num_x), 은닉층의 초기값(num_yh), 출력층의 초기값(num_yo), bias
        self.num_x = num_x + bias
        self.num_yh = num_yh
        self.num_yo = num_yo

        # 활성화 함수 초기값
        self.activation_input = [1.0] * self.num_x
        self.activation_hidden = [1.0] * self.num_yh
        self.activation_out = [1.0] * self.num_yo

        # 가중치 입력 초기값
        self.weight_in = makeMatrix(self.num_x, self.num_yh)
        for i in range(self.num_x):
            for j in range(self.num_yh):
                self.weight_in[i][j] = random.random()

        # 가중치 출력 초기값
        self.weight_out = makeMatrix(self.num_yh, self.num_yo)
        for i in range(self.num_yh):
            for j in range(self.num_yo):
                self.weight_out[i][j] = random.random()

        self.gradient_in = makeMatrix(self.num_x, self.num_yh)
        self.gradient_out = makeMatrix(self.num_yh, self.num_yo)

    def update(self, inputs):
        # 입력층의 활성화 함수
        for i in range(self.num_x - 1):
            self.activation_input[i] = inputs[i]

        # 은닉층의 활성화 함수
        for j in range(self.num_yh):
            sum = 0.0
            for i in range(self.num_x):
                sum += self.activation_input[i] * self.weight_in[i][j]

            # 시그모이드와 tanh 중에서 활성화 함수 선택
            self.activation_hidden[j] = tanh(sum, False)

        # 출력층의 활성화 함수
        for k in range(self.num_yo):
            sum = 0.0
            for j in range(self.num_yh):
                sum += self.activation_hidden[j] * self.weight_out[j][k]

            # 시그모이드와 tanh중에서 활성화 함수 선택
            self.activation_out[k] = tanh(sum, False)

        return self.activation_out[:]

    def backPropagate(self, targets):
        output_deltas = [0.0] * self.num_yo
        # 델타 출력 계산
        for k in range(self.num_yo):
            error = targets[k] - self.activation_out[k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            output_deltas[k] = tanh(self.activation_out[k], True) * error

        # 은닉 노드의 오차 함수
        hidden_deltas = [0.0] * self.num_yh
        for j in range(self.num_yh):
            error = 0.0
            for k in range(self.num_yo):
                error += output_deltas[k] * self.weight_out[j][k]
            # 시그모이드와 tanh 중에서 활성화 함수 선택, 미분 적용
            hidden_deltas[j] = tanh(self.activation_hidden[j], True) * error

        # 출력 가중치 업데이트
        for j in range(self.num_yh):
            for k in range(self.num_yo):
                gradient = output_deltas[k] * self.activation_hidden[j]
                v = mo * self.gradient_out[j][k] - lr * gradient
                self.weight_out[j][k] += v
                self.gradient_out[j][k] = gradient

        # 입력 가중치 업데이트
        for i in range(self.num_x):
            for j in range(self.num_yh):
                gradient = hidden_deltas[j] * self.activation_input[i]
                v = mo * self.gradient_in[i][j] - lr * gradient
                self.weight_in[i][j] += v
                self.gradient_in[i][j] = gradient

        # 오차 계산
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.activation_out[k]) ** 2
        return error

    def train(self, patterns):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                self.backPropagate(targets)
                error += sum(
                    [
                        0.5 * (targets[i] - self.activation_out[i]) ** 2
                        for i in range(len(targets))
                    ]
                )
            if i % 500 == 0:
                print("error: %-.5f" % error)

    # 결과값 출력
    def result(self, patterns):
        for p in patterns:
            print("input: %s, output: %s" % (p[0], self.update(p[0])))


n = NeuralNetwork(2, 2, 1)
n.train(data)
n.result(data)
