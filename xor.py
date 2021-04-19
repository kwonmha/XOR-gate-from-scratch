# -*- coding: utf-8 -*-

import random
import math

inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
answers = [0.0, 1.0, 1.0, 0.0]


def sigmoid(x):
    return 1.0 / (1 + math.exp(-1 * x))


class XorGate(object):
    def __init__(self, learning_rate, num_steps, activation_function):
        self.lr = learning_rate
        self.num_steps = num_steps
        self.activation_func = activation_function

        # random initialize
        self.input_b1 = random.uniform(-1, 1)
        self.input_b2 = random.uniform(-1, 1)
        self.w_ih_11 = random.uniform(-1, 1)
        self.w_ih_12 = random.uniform(-1, 1)
        self.w_ih_21 = random.uniform(-1, 1)
        self.w_ih_22 = random.uniform(-1, 1)

        self.hidden_b = random.uniform(-1, 1)
        self.w_ho_1 = random.uniform(-1, 1)
        self.w_ho_2 = random.uniform(-1, 1)

    def train(self, input_data, answer_data):
        for step in range(self.num_steps):
            sum_error = 0.0
            for inputs, answer in zip(input_data, answer_data):
                output, h1, h2 = self.compute_output(inputs)
                sum_error += 0.5 * (answer - output) ** 2
                self.weight_update(answer, output, h1, h2, inputs)
            mean_squared_error = sum_error / 4.0

            if step % 100 == 0:
                self.predict(input_data)
                print("mean_squared_error:", mean_squared_error)
                print("=============")

    def compute_output(self, input_data):
        # calc hidden layer
        h1_in = self.w_ih_11 * input_data[0] + self.w_ih_21 * input_data[1] + self.input_b1
        h2_in = self.w_ih_12 * input_data[0] + self.w_ih_22 * input_data[1] + self.input_b2

        # activation function
        h1 = self.activation_func(h1_in)
        h2 = self.activation_func(h2_in)

        # calc output, activation function
        output_in = self.w_ho_1 * h1 + self.w_ho_2 * h2 + self.hidden_b
        output = self.activation_func(output_in)

        return output, h1, h2

    def weight_update(self, answer, output, h1, h2, input_data):
        #       (d MSE/d output) * (d sigmoid / d output_in)
        delta = (answer - output) * output * (1 - output)

        # update this parameters before w_ho are updated.
        # w, b(t+1) = w, b + learning rate * dE / dw,b
        #                                             (d output_in / d h) * (d sigmoid/ d h_in) * (d h_in / d w_ih)
        self.w_ih_11 = self.w_ih_11 + self.lr * delta * self.w_ho_1 * h1 * (1 - h1) * input_data[0]
        self.w_ih_12 = self.w_ih_12 + self.lr * delta * self.w_ho_2 * h2 * (1 - h2) * input_data[0]
        self.w_ih_21 = self.w_ih_21 + self.lr * delta * self.w_ho_1 * h1 * (1 - h1) * input_data[1]
        self.w_ih_22 = self.w_ih_22 + self.lr * delta * self.w_ho_2 * h2 * (1 - h2) * input_data[1]
        self.input_b1 = self.input_b1 + self.lr * delta * self.w_ho_1 * h1 * (1 - h1)
        self.input_b2 = self.input_b2 + self.lr * delta * self.w_ho_2 * h2 * (1 - h2)

        #                                        (d output_in / d w_ho)
        self.w_ho_1 = self.w_ho_1 + self.lr * delta * h1
        self.w_ho_2 = self.w_ho_2 + self.lr * delta * h2
        self.hidden_b = self.hidden_b + self.lr * delta

    def predict(self, input_data):
        for inputs in input_data:
            print(inputs[0], inputs[1], self.compute_output(inputs)[0])


model = XorGate(learning_rate=5.0, num_steps=2000, activation_function=sigmoid)
model.train(inputs, answers)
print("final result:")
model.predict(inputs)

