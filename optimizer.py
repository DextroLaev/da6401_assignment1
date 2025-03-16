import numpy as np
import math
from abc import abstractmethod
from typing import List,Tuple

class Optimizer:
	def __init__(self, learning_rate: float, **kwargs) -> None:

		"""
			Base class for optimizers.

			Args:

				Learning_rate (float): learning_rate for optimizer
		"""
		self.learning_rate = learning_rate

	@abstractmethod
	def update(self, W: List[np.ndarray], B: List[np.ndarray],
		dw: List[np.ndarray], db: List[np.ndarray]) -> tuple[List[np.ndarray],List[np.ndarray]]:
		"""
	Abstract method to apply optimizer-specific update rule

	Args:
		W: List of weight matrices
		B: List of bias vectors
		dw: List of weight gradients
		db: List of bias gradients

	Returns:
		Tuple[List[np.ndarray],List[np.ndarray]]: updated weights and biases
	"""
		pass

	@abstractmethod
	def config(self,W,B) -> None:
		'adds the neccesary stuffs needed for specific optimizer'
		pass

class SGD(Optimizer):
	def update(self,W,B,dw,db):
		for i in range(len(W)):
			W[i] -= self.learning_rate * dw[i]
			B[i] -= self.learning_rate * db[i]

class Momentum(Optimizer):
	def __init__(self,learning_rate,**kwargs):
		super().__init__(learning_rate)		
		self.beta = kwargs['momentum']
		self.momentum_W = None
		self.momentum_B = None


	def config(self,W,B):
		self.optimizer_config = {}
		self.optimizer_config['momentum_W'] = [np.zeros_like(w) for w in W]
		self.optimizer_config['momentum_B'] = [np.zeros_like(b) for b in B]


	def update(self,W,B,dw,db):
		momentum_W = self.optimizer_config['momentum_W']
		momentum_B = self.optimizer_config['momentum_B']
		for i in range(len(W)):
			momentum_W[i] = self.beta * momentum_W[i] + (1-self.beta)*dw[i]
			momentum_B[i] = self.beta * momentum_B[i] + (1-self.beta)*db[i]

			W[i] -= self.learning_rate * momentum_W[i]
			B[i] -= self.learning_rate * momentum_B[i]

		self.optimizer_config['momentum_W'] = momentum_W
		self.optimizer_config['momentum_B'] = momentum_B

class Nestrov(Momentum):
	def __init__(self,learning_rate,**kwargs):
		super().__init__(learning_rate, **kwargs)
		self.beta = kwargs['momentum']


	def config(self,W,B):
		self.optimizer_config = {}
		self.optimizer_config['momentum_W'] = [np.zeros_like(w) for w in W]
		self.optimizer_config['momentum_B'] = [np.zeros_like(b) for b in B]

	def update(self, W, B, dw, db):
		for i in range(len(W)):
			self.optimizer_config['momentum_W'][i] = self.beta * self.optimizer_config['momentum_W'][i] + self.learning_rate* dw[i]
			self.optimizer_config['momentum_B'][i] = self.beta * self.optimizer_config['momentum_B'][i] + self.learning_rate* db[i]
			# W[i] -= self.learning_rate * self.optimizer_config['momentum_W'][i]
			# B[i] -= self.learning_rate * self.optimizer_config['momentum_B'][i]
			W[i] -= self.optimizer_config['momentum_W'][i]
			B[i] -= self.optimizer_config['momentum_B'][i]
        

class RMSProp(Optimizer):
    def __init__(self, learning_rate, **kwargs):
        super().__init__(learning_rate)
        self.beta = kwargs['beta']
        self.eps = kwargs['eps']

    def config(self,W,B):
    	self.optimizer_config = {}
    	self.optimizer_config['v_W'] = [np.zeros_like(w) for w in W]
    	self.optimizer_config['v_B'] = [np.zeros_like(b) for b in B]

    def update(self, W, B, dw, db):
        
        for i in range(len(W)):
            self.optimizer_config['v_W'][i] = self.beta * self.optimizer_config['v_W'][i] + (1 - self.beta) * (dw[i] ** 2)
            self.optimizer_config['v_B'][i] = self.beta * self.optimizer_config['v_B'][i] + (1 - self.beta) * (db[i] ** 2)

            adaptive_lr_w = self.learning_rate/(np.sqrt(self.optimizer_config['v_W'][i]+self.eps))
            adaptive_lr_b = self.learning_rate/(np.sqrt(self.optimizer_config['v_B'][i]+self.eps))
            W[i] -= adaptive_lr_w * dw[i]
            B[i] -= adaptive_lr_b * db[i]


        

class Adam(Optimizer):
    def __init__(self, learning_rate, **kwargs):
        super().__init__(learning_rate)

        self.beta1 = kwargs['beta1']
        self.beta2 = kwargs['beta2']
        self.eps = kwargs['eps']
        self.t = 0
        
    def config(self,W,B):
    	self.optimizer_config = {}
    	self.optimizer_config['momentum1_W'] = [np.zeros_like(w) for w in W]
    	self.optimizer_config['momentum1_B'] = [np.zeros_like(b) for b in B]
    	self.optimizer_config['momentum2_W'] = [np.zeros_like(w) for w in W]
    	self.optimizer_config['momentum2_B'] = [np.zeros_like(b) for b in B]
    	self.optimizer_config['t'] = 0

    def update(self, W, B, dw, db):
        
        self.t += 1
        for i in range(len(W)):
            self.optimizer_config['momentum1_W'][i] = self.beta1 * self.optimizer_config['momentum1_W'][i] + (1 - self.beta1) * dw[i]
            self.optimizer_config['momentum1_B'][i] = self.beta2 * self.optimizer_config['momentum1_B'][i] + (1 - self.beta2) * db[i]

            self.optimizer_config['momentum2_W'][i] = self.beta1 * self.optimizer_config['momentum2_W'][i] + (1 - self.beta1) * (dw[i]**2)
            self.optimizer_config['momentum2_B'][i] = self.beta2 * self.optimizer_config['momentum2_B'][i] + (1 - self.beta2) * (db[i]**2)

            momentum1_W_hat = self.optimizer_config['momentum1_W'][i]/(1-(self.beta1**self.t))
            momentum1_B_hat = self.optimizer_config['momentum1_B'][i]/(1-(self.beta1**self.t))

            momentum2_W_hat = self.optimizer_config['momentum2_W'][i]/(1-(self.beta2**self.t))
            momentum2_B_hat = self.optimizer_config['momentum2_B'][i]/(1-(self.beta2**self.t))

            adaptive_lr_W = self.learning_rate/(np.sqrt(momentum2_W_hat) + self.eps)
            adaptive_lr_B = self.learning_rate/(np.sqrt(momentum2_B_hat) + self.eps)
            W[i] -= adaptive_lr_W * momentum1_W_hat
            B[i] -= adaptive_lr_B * momentum1_B_hat

class Nadam(Optimizer):
    def __init__(self, learning_rate,**kwargs):
        super().__init__(learning_rate)
        self.beta1 = kwargs['beta1']
        self.beta2 = kwargs['beta2']
        self.eps = kwargs['eps']
        self.t = 0
        
    def config(self,W,B):
    	self.optimizer_config = {}
    	self.optimizer_config['momentum1_W'] = [np.zeros_like(w) for w in W]
    	self.optimizer_config['momentum1_B'] = [np.zeros_like(b) for b in B]
    	self.optimizer_config['momentum2_W'] = [np.zeros_like(w) for w in W]
    	self.optimizer_config['momentum2_B'] = [np.zeros_like(b) for b in B]

    def update(self, W, B, dw, db):
        
        self.t += 1
        for i in range(len(W)):
            self.optimizer_config['momentum1_W'][i] = self.beta1 * self.optimizer_config['momentum1_W'][i] + (1 - self.beta1) * dw[i]
            self.optimizer_config['momentum1_B'][i] = self.beta2 * self.optimizer_config['momentum1_B'][i] + (1 - self.beta2) * db[i]

            self.optimizer_config['momentum2_W'][i] = self.beta1 * self.optimizer_config['momentum2_W'][i] + (1 - self.beta1) * (dw[i]**2)
            self.optimizer_config['momentum2_B'][i] = self.beta2 * self.optimizer_config['momentum2_B'][i] + (1 - self.beta2) * (db[i]**2)

            momentum1_W_hat = self.optimizer_config['momentum1_W'][i]/(1-(self.beta1**self.t))
            momentum1_B_hat = self.optimizer_config['momentum1_B'][i]/(1-(self.beta1**self.t))

            momentum2_W_hat = self.optimizer_config['momentum2_W'][i]/(1-(self.beta2**self.t))
            momentum2_B_hat = self.optimizer_config['momentum2_B'][i]/(1-(self.beta2**self.t))

            m_nestrov_W = self.beta1 * momentum1_W_hat + ((1 - self.beta1) * dw[i])/(1-self.beta1**self.t)
            m_nestrov_B = self.beta1 * momentum1_B_hat + ((1 - self.beta1) * db[i])/(1-self.beta1**self.t)

            adaptive_lr_W = self.learning_rate / (np.sqrt(momentum2_W_hat) + self.eps)
            adaptive_lr_B = self.learning_rate / (np.sqrt(momentum2_B_hat) + self.eps)
            W[i] -= adaptive_lr_W * m_nestrov_W
            B[i] -= adaptive_lr_B * m_nestrov_B


