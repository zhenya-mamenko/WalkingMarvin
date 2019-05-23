import sys

only_walking = False
track_weights = False
detailed_log = False
load_weights_file = ""
save_weights_file = ""
max_episodes = 0

print("\nWalking Marving by jtahirov & emamenko (Kukuruzka team)")
for i in range(1, len(sys.argv)):
	if (sys.argv[i] == "-h" or sys.argv[i] == "--help"):
		print("\n",
			"  –-walk (-w)\t\tDisplay only walking process.\n",
			"  –-help (-h)\t\tDisplay available commands.\n",
			"  –-load (-l) file\tLoad weights for Marvin agent from a file.\n\t\t\tSkip training process if this option is specified.\n",
			"  –-save (-s) file\tSave weights to a file after running the program.\n",
			"  --track (-t)\t\tSave weights after each 10 episodes.\n",
			"  –-detailed-log (-d)\tShow weights after each episode.\n",
			"  –-episodes (-e) num\tMaximum episodes in learning process.\n\t\t\tDefault - until get episode's reward >= 100.\n"
			)
		exit()
	if (sys.argv[i] == "-w" or sys.argv[i] == "--walk"):
		only_walking = True
	if (sys.argv[i] == "-l" or sys.argv[i] == "--load"):
		i += 1
		load_weights_file = sys.argv[i]
	if (sys.argv[i] == "-s" or sys.argv[i] == "--save"):
		i += 1
		save_weights_file = sys.argv[i]
	if (sys.argv[i] == "-t" or sys.argv[i] == "--track"):
		track_weights = True
	if (sys.argv[i] == "-d" or sys.argv[i] == "--detailed-log"):
		detailed_log = True
	if (sys.argv[i] == "-e" or sys.argv[i] == "--episodes"):
		i += 1
		max_episodes = (int)(sys.argv[i])

import gym
import math
import random
import numpy as np
import matplotlib
import pandas as pd

env = gym.make('Marvin-v0')


# Neuron class

def mult_and_sum(a, b):
	result = 0
	for i in range(len(a)):
		result += a[i] * b[i]
	return result

class Neuron(object):
	def __init__(self, input_counts = 1, activation_function = math.tanh, weights = None,
				update_function=None, learning_rate = 0.001, input_layer = False, value = None):
		self.input_layer = input_layer
		if self.input_layer == False:
			self.weights = [random.uniform(-1, 1)] * input_counts if weights == None else weights
		self.value = value
		self.activation_function = activation_function
		self.update_function = self.update_weights if update_function == None else update_function
		self.learning_rate = learning_rate

	def setValue(self, value):
		self.value = value

	def activate(self, inputs = None):
		if (self.input_layer):
			return self.value
		self.setValue(self.activation_function(mult_and_sum(inputs, self.weights)))
		return self.value

	def update_weights(self, inp, update):
		errors = self.learning_rate * (update - self.activate(inp))
		self.weights = np.add(self.weights, np.multiply(inp, errors))


# Layer class

def linear_classifiyer(x):
	return x

class Layer(object):
	def __init__(self, neuron_count, input_counts = 1, neuron_weights = None,
				 activation_function = linear_classifiyer):
		random.seed()
		self.neurons = []
		self.input_layer = True if input_counts == 1 else False

		for i in range(neuron_count):
			if (neuron_weights != None):
				self.neurons.append(Neuron(input_counts, input_layer = self.input_layer, weights = neuron_weights[i]))
			else:
				self.neurons.append(Neuron(input_counts, input_layer = self.input_layer))

	def setValues(self, inputs):
		for index, neuron in enumerate(self.neurons):
			neuron.setValue(inputs[index])

	def update(self, inputs, reward):
		if (self.input_layer):
			return
		for neuron in self.neurons:
			neuron.update_weights(inputs, reward)

	def output(self, inputs = None):
		layer_output = []
		for neuron in self.neurons:
			layer_output.append(neuron.activate(inputs))
		return layer_output


# Brain class

class Brain(object):
	def __init__(self, input_counts, weights):
		self.layers = []
		self.input_layer = Layer(neuron_count = input_counts)
		self.layers.append(self.input_layer)
		for i, layerWeight in enumerate(weights):
			layer = layerWeight
			self.layers.append(Layer(neuron_count = len(layer), input_counts = len(layer[-1]),
									 neuron_weights = layer, activation_function = math.tanh))
		self.output_layer = self.layers[-1]

	def generate_action(self, data):
		self.input_layer.setValues(data)
		inputs  = self.input_layer.output()
		for hidden_layer in self.layers[1: -1]:
			inputs = hidden_layer.output(inputs)
		return self.output_layer.output(inputs)

	def learn(self, observation, action, reward):
		data = np.concatenate((observation, action), axis=None)
		self.input_layer.setValues(data)
		layer_input = self.input_layer.output()
		for layer in self.layers[1:]:
			layer.update(layer_input, reward)
			layer_input = layer.output(layer_input)


# Support functions

def initializeRandomWeights(schema):
	weights = []
	for layer in schema:
		neurons = []
		for neuron in layer:
			neuront = []
			for k in range(neuron[0]):
				neuront.append(random.uniform(-1, 1))
			neurons.append(neuront)
		weights.append(neurons)
	return weights

def createSchema(layers = 1, neurons = [1]):
	schema = []
	layer_len = neurons[0]
	for n in neurons:
		layer = []

		for _ in range(n):
			layer.append([layer_len])
		layer_len = len(layer)
		schema.append(layer)
	return schema

def convertWeightsToNumpy(weights):
	w = []
	for layer in weights:
		for neuron in layer:
			for element in neuron:
				w.append(element)
	return np.array(w)

def convertWeightsToList(schema, w):
	weights = []
	i = 0
	for layer in schema:
		neurons = []
		for neuron in layer:
			neuront = []
			for k in range(neuron[0]):
				neuront.append(w[i])
				i = i + 1
			neurons.append(neuront)
		weights.append(neurons)
	return weights


# Evaluate function

def f(w):
	observation = env.reset()
	tester = Brain(input_counts = 24, weights = w)
	total = 0
	for i in range(350):
		action = tester.generate_action(observation)
		new_observation, reward, done, info = env.step(action)
		if (np.std(new_observation - observation) < 0.00001):
			reward = -100
		observation = new_observation
		total += reward
		if (done == True or reward == -100 or total > 100):
			break
	env.close()
	return (total, i + 1)


# Marvin

schema = createSchema(layers=3, neurons = [24, 6, 4])

if (load_weights_file != ""):
	data = pd.read_csv(load_weights_file)
	w = np.array(data["weight"])
else:
	w = convertWeightsToNumpy(initializeRandomWeights(schema))

# Learning

if (only_walking == False):
	print("\nLearning:")
	npop = 50 # population size
	sigma = 0.1 # noise standard deviation
	alpha = 0.05 # learning rate
	np.random.seed(422142)

	i = 0
	while (True):
		steps = 0
		reward = 0
		N = np.random.random_sample((npop, w.shape[0]))
		R = np.zeros(npop)

		for j in range(npop):
			w_try = w + sigma * N[j]
			R[j], s = f(convertWeightsToList(schema, w_try))
			steps += s
			reward += R[j]

		A = (R - np.mean(R)) / np.std(R)
		w = w + alpha / (npop * sigma) * np.dot(N.T, A)

		print('Episode: %d.\tSteps: %.1f.\tReward: %.1f' % (i + 1, steps / npop, reward / npop))
		if (detailed_log == True):
			print("Weights: ", str(w))

		i += 1
		if (i % 10 == 0 and track_weights == True and save_weights_file != ""):
			data = pd.DataFrame(w, columns=['weight'])
			data.to_csv(save_weights_file)

		if (reward / npop > 100 or (max_episodes != 0 and max_episodes == i)):
			break

if (save_weights_file != ""):
	data = pd.DataFrame(w, columns=['weight'])
	data.to_csv(save_weights_file)


# Walking

print("\nWalking:")
weights = convertWeightsToList(schema, w)
marvin = Brain(input_counts = 24, weights = weights)
total = 0
observation = env.reset()
for step in range(1500):
	env.render()
	action = marvin.generate_action(observation)
	observation, reward, done, info = env.step(action)
	total += reward
	if (done == True or reward == -100):
		break
env.close()

print('Steps: %d.\tTotal reward: %.1f\n' % (step + 1, total))
