import numpy as np
import math
from abc import abstractmethod
from typing import List,Tuple

class Optimizer:
	def __init__(self, learning_rate: float) -> None:

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
	def config(self) -> None:
		pass


class SGD(Optimizer):
	def update(self,W,B,dw,db):
		for i in range(len(W)):
			W[i] -= self.learning_rate * dw[i]
			B[i] -= self.learning_rate * db[i]

		return W,B

# class Momentum(Optimizer):
# 	def __init__(self,learning_rate,beta=0.9):
# 		super().__init__(learning_rate)
# 		self.beta = beta
# 		self.momentum_W = None
# 		self.momentum_B = None
# 		self.initialize_momentum(W,B)

# 	def config(self,W,B):
# 		optimizer_config = {'learning_rate':learning_rate,**kwargs}
# 		momentum_W = [np.zeros_like(w) for w in W]
#         momentum_B = [np.zeros_like(b) for b in B]
#         optimizer_config['momentum_W'] = momentum_W
#         optimizer_config['momentum_B'] = momentum_B

# 	def initialize_momentum(self,W,B) -> None:
# 		if self.momentum_W is None:
# 			self.momentum_W = [np.zeros_like(w) for w in W]
# 			self.momentum_B = [np.zeros_like(b) for b in B]

# 	def update(self,W,B,dw,db):
		
# 		for i in range(len(W)):
# 			self.momentum_W[i] = self.beta * self.momentum_W[i] + (1-self.beta)*dw[i]
# 			self.momentum_B[i] = self.beta * self.momentum_B[i] + (1-self.beta)*db[i]

# 			W[i] -= self.learning_rate * self.momentum_W[i]
# 			B[i] -= self.learning_rate * self.momentum_B[i]

# 		return W,B

# class Nesterov(Momentum):

# 	def config(self,W,B):
# 		momentum_W = [np.zeros_like(w) for w in self.W]
#         momentum_B = [np.zeros_like(b) for b in self.B]
#         optimizer_config['momentum_W'] = momentum_W
#         optimizer_config['momentum_B'] = momentum_B

#     def update(self, W, B, dw, db):
#         # self.initialize_momentum(W, B)
#         lookahead_W = [w - self.beta * v for w, v in zip(W, self.momentum_W)]
#         lookahead_B = [b - self.beta * v for b, v in zip(B, self.momentum_B)]

#         for i in range(len(W)):
#             self.momentum_W[i] = self.beta * self.momentum_W[i] + dw[i]
#             self.momentum_B[i] = self.beta * self.momentum_B[i] + db[i]

#             W[i] -= self.learning_rate * self.momentum_W[i]
#             B[i] -= self.learning_rate * self.momentum_B[i]

#         return W, B

# class RMSProp(Optimizer):
#     def __init__(self, learning_rate, beta=0.9, eps=1e-8):
#         super().__init__(learning_rate)
#         self.beta = beta
#         self.eps = eps
#         self.v_W = None
#         self.v_B = None
#         self.initialize_cache(W, B)

#     def config(self,W,B):
#     	v_W = [np.zeros_like(w) for w in self.W]
#         v_B = [np.zeros_like(b) for b in self.B]
#         optimizer_config['v_W'] = v_W
#         optimizer_config['v_B'] = v_B

#     def initialize_cache(self, W, B):
#         if self.v_W is None:
#             self.v_W = [np.zeros_like(w) for w in W]
#             self.v_B = [np.zeros_like(b) for b in B]

#     def update(self, W, B, dw, db):
        
#         for i in range(len(W)):
#             self.v_W[i] = self.beta * self.v_W[i] + (1 - self.beta) * (dw[i] ** 2)
#             self.v_B[i] = self.beta * self.v_B[i] + (1 - self.beta) * (db[i] ** 2)

#             W[i] -= self.learning_rate * dw[i] / (np.sqrt(self.v_W[i]) + self.eps)
#             B[i] -= self.learning_rate * db[i] / (np.sqrt(self.v_B[i]) + self.eps)
#         return W, B

# class Adam(Optimizer):
#     def __init__(self, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
#         super().__init__(learning_rate)
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.m_W = None
#         self.v_W = None
#         self.m_B = None
#         self.v_B = None
#         self.t = 0
#         self.initialize_cache(W, B)

#     def config(self,W,B):
#     	momentum1_W = [np.zeros_like(w) for w in W]
#         momentum1_B = [np.zeros_like(b) for b in B]
#         momentum2_W = [np.zeros_like(w) for w in W]
#         momentum2_B = [np.zeros_like(b) for b in B]
#         optimizer_config['momentum1_W'] = momentum1_W
#         optimizer_config['momentum1_B'] = momentum1_B
#         optimizer_config['momentum2_W'] = momentum2_W
#         optimizer_config['momentum2_B'] = momentum2_B
#         optimizer_config['t'] = 0

#     def initialize_cache(self, W, B):
#         if self.m_W is None:
#             self.m_W = [np.zeros_like(w) for w in W]
#             self.v_W = [np.zeros_like(w) for w in W]
#             self.m_B = [np.zeros_like(b) for b in B]
#             self.v_B = [np.zeros_like(b) for b in B]

#     def update(self, W, B, dw, db):
        
#         self.t += 1
#         for i in range(len(W)):
#             self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dw[i]
#             self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dw[i] ** 2)

#             m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
#             v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)

#             W[i] -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.eps)

#             self.m_B[i] = self.beta1 * self.m_B[i] + (1 - self.beta1) * db[i]
#             self.v_B[i] = self.beta2 * self.v_B[i] + (1 - self.beta2) * (db[i] ** 2)

#             m_B_hat = self.m_B[i] / (1 - self.beta1 ** self.t)
#             v_B_hat = self.v_B[i] / (1 - self.beta2 ** self.t)

#             B[i] -= self.learning_rate * m_B_hat / (np.sqrt(v_B_hat) + self.eps)

#         return W, B
