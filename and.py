# -*- coding: utf-8 -*-

import random

inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
answers = [0.0, 0.0, 0.0, 1.0]


class AndGate(object):
    def __init__(self, learning_rate, num_steps):
        self.lr = learning_rate
        self.num_steps = num_steps

        # random initialize
        self.b = random.uniform(-1, 1)
        self.w1 = random.uniform(-1, 1)
        self.w2 = random.uniform(-1, 1)

    def train(self, input_data, answer_data):
        for step in range(self.num_steps):
            error = 0.0
            for inputs, answer in zip(input_data, answer_data):
                output = self.compute_output(inputs)
                error += 0.5 * (answer - output) ** 2
                self.weight_update(answer, output, inputs)
            mean_squared_error = error / 4.0

            if step % 10 == 0:
                self.predict(input_data)
                print("mean_squared_error:", mean_squared_error)
                print("=============")

    def compute_output(self, input_data):
        output = input_data[0] * self.w1 + input_data[1] * self.w2 + self.b
        return output

    def weight_update(self, answer, output, input_data):
        self.w1 = self.w1 + self.lr * (answer - output) * input_data[0]
        self.w2 = self.w2 + self.lr * (answer - output) * input_data[1]
        self.b = self.b + self.lr * (answer - output)

    def predict(self, input_data):
        for inputs in input_data:
            print(inputs[0], inputs[1], self.compute_output(inputs))


model = AndGate(learning_rate=0.01, num_steps=2000)
model.train(inputs, answers)
print("final result:")
model.predict(inputs)

