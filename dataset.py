from keras.datasets import fashion_mnist,mnist
import numpy as np

class Dataset:
	"""
	A class to load and managet datasets.

	Attributes:
		dataset_name (str): The name of the dataset
	"""

	def __init__(self, dataset_name:str) -> None:

		"""Args:
				dataset_name (str)
		"""

		if dataset_name not in {'fashion_mnist','mnist'}:
			raise ValueError("Invalid dataset name, suuported values are 'mnist','fashion_mnist'")

		self.dataset_name = dataset_name

	def load_data(self) -> tuple[tuple,tuple,tuple]:
		"""

		Return format:
			((train_data,train_label),(test_data,test_label),(val_data,val_label))
		"""

		if self.dataset_name == 'mnist':
			# (train_data,train_label),(test_data,test_label) = mnist.load_data()
			dataset = mnist.load_data()
		elif self.dataset_name == 'fashion_mnist':
			# (train_data,train_label),(test_data,test_label) = fashion_mnist.load_data()
			dataset = fashion_mnist.load_data()

		return self.preprocess_data(dataset)
		

	def preprocess_data(self,dataset:tuple[tuple,tuple]) -> tuple[tuple,tuple,tuple]:
		"""
			Processing done:
				1. Shuffling the dataset.
				2. Reshaping the Image data.
				3. Normalization of the data.
				4. Converted all the labels to one-hot encoded labels
				5. Converted 10% of training data to validation data

			Return Format:
				((train_data,train_label),(test_data,test_label),(val_data,val_label))
		"""

		(train_data,train_label),(test_data,test_label) = dataset
		train_indices = np.arange(train_data.shape[0])
		np.random.shuffle(train_indices)
		train_data = train_data[train_indices]
		train_label = train_label[train_indices]

		# Shuffle test data
		test_indices = np.arange(test_data.shape[0])
		np.random.shuffle(test_indices)
		test_data = test_data[test_indices]
		test_label = test_label[test_indices]

		# reshaping the image of shape (28x28) -> (1,784)
		train_data = train_data.reshape(train_data.shape[0],-1)
		test_data = test_data.reshape(test_data.shape[0],-1)

		# normalizing the data
		train_data = train_data/255.0
		test_data = test_data/255.0

		# converting the label data to one-hot encoded data
		train_label = np.eye(10)[train_label]
		test_label = np.eye(10)[test_label]

		# splitting 10% of training data as validation data
		total_train_data = len(train_data)
		split_index = int(0.9 * total_train_data)
		train_data, val_data = train_data[:split_index], train_data[split_index:]
		train_label, val_label = train_label[:split_index], train_label[split_index:]

		return (train_data,train_label),(test_data,test_label),(val_data,val_label)

if __name__ == '__main__':
	dataset = Dataset('mnist')
	(train_data,train_label),(test_data,test_label),(val_data,val_labels) = dataset.load_data()
	print(train_data.shape)
	print(test_data.shape)
	print(val_data.shape)