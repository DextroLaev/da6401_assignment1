import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist

def load_dataset():
	(train_data,train_label),(test_data,test_label) = fashion_mnist.load_data()
	return train_data,train_label,test_data,test_label

def plot_img_wandb(data,label):
	class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]
	each_class_img_index = []
	classes = []
	for i in range(len(label)):
		if label[i] not in classes:
			each_class_img_index.append(i)
			classes.append(label[i])
		if len(each_class_img_index) == 10:
			break
	images = []
	wandb.init(project='dl-assignment1')
	# for idx in each_class_img_index:
	#     img = data[idx]
	#     img_label = class_names[label[idx]]
	#     images.append(wandb.Image(img, caption=img_label))
	#     wandb.log({'image':wandb.Image(img,caption=img_label)})

	wandb.log({'Image':[wandb.Image(img,caption=img_label) for img, img_label
		in zip(data,classes)]})
	# wandb.log({"Fashion-MNIST Grid": images})
	wandb.finish()
	

if __name__ == '__main__':
	wandb.login()
	train_data,train_label,test_data,test_label = load_dataset()
	plot_img_wandb(train_data,train_label)