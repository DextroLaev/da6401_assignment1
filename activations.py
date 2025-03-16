import numpy as np
from abc import abstractmethod

class Activation:

	@abstractmethod
	def out(self,x):
		pass

	@abstractmethod
	def derivative(self,x):
		pass


class Sigmoid(Activation):

	def out(self,x):
		return 1/(1+np.exp(-x))

	def derivative(self,x):
		return self.out(x)*(1-self.out(x))

class ReLU(Activation):

	def out(self,x):
		return np.maximum(0,x)

	def derivative(self,x):
		return np.where(x > 0, 1, 0)

class Tanh(Activation):

	def out(self,x):
		return np.tanh(x)

	def derivative(self,x):
		return 1 - np.tanh(x)**2

class Softmax(Activation):

	def out(self,x):
		# if np.isnan(x).any() or np.isinf(x).any():
		# 	x = np.nan_to_num(x)
		exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
		out_val = exp_x / np.sum(exp_x, axis=1, keepdims=True)
		return out_val

	def derivative(self,x):
		pass

class Linear(Activation):

	def out(self,x):
		return x

	def derivative(self,x):
		return 1
