import numpy as np
import math

class Loss:

	def compute(self, *args) -> float:
		pass

class CrossEntropyLoss(Loss):
	"""
		calculates the cross-entropy loss
	"""

	def compute(self, pred_logits: np.ndarray, true_class: np.ndarray) -> float:
		
		eps = 1e-15
		pred_logits = np.clip(pred_logits,eps,1-eps)
		loss_val = - np.sum(true_class*np.log(pred_logits+1e-18),axis=1)
		return np.mean(loss_val)


class MeanSquaredErrorLoss(Loss):
	"""
		calculates the mean-squared error loss
	"""

	def compute(self,pred_logits: np.ndarray, true_class: np.ndarray) -> float:

		loss_val = np.mean(np.square(pred_logits - true_class))
		return loss_val