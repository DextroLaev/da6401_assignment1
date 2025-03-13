import numpy as np
from loss import *
from dataset import *
# from optimizers import *
from optimizer2 import SGD,Momentum
from typing import List,Tuple
import sys

class Neural_Net:

    def __init__(self,input_shape,number_of_hidden_layers,hidden_neurons_per_layer,activation_function,output_shape,type_of_init,L2reg_const) -> None:
        self.input_shape = input_shape
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_neurons_per_layer = hidden_neurons_per_layer
        self.output_shape = output_shape
        self.activation_function = activation_function
        self.type_of_init = type_of_init
        self.L2reg_const = L2reg_const
        self.W, self.B = self.weight_init(type_of_init)

    def weight_init(self,type_of_init: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        W = []
        B = []
        
        if type_of_init not in {'Xavier','random'}:
        	raise ValueError("Given initialization doesn't exist, choose from 'Xavier','random'")

        if type_of_init == 'Xavier':    
            W.append(np.random.randn(self.input_shape, self.hidden_neurons_per_layer[0]) * np.sqrt(6 / (self.input_shape + self.hidden_neurons_per_layer[0])))
            for i in range(1, self.number_of_hidden_layers):
                W.append(np.random.randn(self.hidden_neurons_per_layer[i-1], self.hidden_neurons_per_layer[i]) *np.sqrt(6 / (self.hidden_neurons_per_layer[i-1] + self.hidden_neurons_per_layer[i])))
            W.append(np.random.randn(self.hidden_neurons_per_layer[-1], self.output_shape)*np.sqrt(6 / (self.hidden_neurons_per_layer[-1] + self.output_shape)))

        elif type_of_init == 'random':            
            W.append(np.random.rand(self.input_shape,self.hidden_neurons_per_layer[0])-0.5)
            for i in range(1,self.number_of_hidden_layers):
                W.append(np.random.rand(self.hidden_neurons_per_layer[i-1],self.hidden_neurons_per_layer[i])-0.5)
            W.append(np.random.rand(self.hidden_neurons_per_layer[-1],self.output_shape)-0.5)
            
        B.append(np.random.rand(1,self.hidden_neurons_per_layer[0])-0.5)
        for i in range(1,self.number_of_hidden_layers):
            B.append(np.random.rand(1,self.hidden_neurons_per_layer[i])-0.5)
        
        B.append(np.random.rand(1,self.output_shape)-0.5)
        return W,B


    def feed_forward(self,data: np.ndarray, W: List[np.ndarray], B: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    	A = []         #pre-activation output list
    	H = []         #post-ativation output list
    	input_d = data
    	for i in range(self.number_of_hidden_layers):
    		linear_out = np.dot(input_d,W[i]) + B[i]
    		activation_out = self.activation(linear_out)
    		A.append(linear_out)
    		H.append(activation_out)
    		input_d = activation_out
    	y_pred = self.softmax(np.dot(H[-1],W[-1]) + B[-1])
    	return A,H,y_pred

    def activation(self,x:str):
    	if self.activation_function == 'ReLU':
    		return self.ReLU(x)
    	elif self.activation_function == 'tanh':
    		return self.tanh(x)
    	elif self.activation_function == 'sigmoid':
    		return self.sigmoid(x)
    	elif self.activation_function == 'linear':
    		return x

    def activation_derivative(self,x:str):
        if self.activation_function == 'ReLU':
            return self.ReLU_derivative(x)
        elif self.activation_function == 'tanh':
            return self.tanh_derivative(x)
        elif self.activation_function == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_function == 'linear':
            return 1

    def ReLU(self,x: List[np.ndarray]):
        return np.maximum(0,x)

    def ReLU_derivative(self,x:List[np.ndarray]):
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self,x: List[np.ndarray]):
        # x = np.clip(x, -500, 500)
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x: List[np.ndarray]):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def tanh(self,x: List[np.ndarray]):
        return np.tanh(x)

    def tanh_derivative(self,x: List[np.ndarray]):
        return 1 - self.tanh(x)**2

    def softmax(self,x: List[np.ndarray]):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return out

    def accuracy(self,data: List[np.ndarray],label:List[np.ndarray]) -> float:
        A,H,y_prob = self.feed_forward(data,self.W,self.B)
        acc = 0
        for i in range(len(data)):
            if np.argmax(label[i]) == np.argmax(y_prob[i]):
                acc += 1
        return acc/data.shape[0]

    def gradients(self,A: List[np.ndarray],H:List[np.ndarray],W:List[np.ndarray],B:List[np.ndarray],
    	batch_data: List[np.ndarray], y_prob: List[np.ndarray], batch_label: List[np.ndarray]):
    	dW = []
    	db = []

    	# Calculate initial error (cross-entropy gradient for softmax)
    	error = y_prob - batch_label
    	# Backpropagate through output layer (softmax)
    	dW.insert(0, np.dot(H[-1].T, error)+self.L2reg_const * W[-1])
    	db.insert(0, np.sum(error,axis=0,keepdims=True))

    	# Backpropagate through hidden layers
    	delta = error
    	for i in range(self.number_of_hidden_layers - 1, 0, -1):
    		delta = np.dot(delta, W[i+1].T) * self.activation_derivative(A[i])
    		dW.insert(0, np.dot(H[i-1].T, delta)+ self.L2reg_const * W[i])
    		db.insert(0, np.sum(delta, axis=0,keepdims=True))

    	# Backpropagate through input layer
    	delta = np.dot(delta, W[1].T) * self.activation_derivative(A[0])
    	dW.insert(0, np.dot(batch_data.T, delta)+ self.L2reg_const * W[0])
    	db.insert(0, np.sum(delta, axis=0,keepdims=True))
    	return dW, db

    def train(self,optimizer:str,epochs:int,learning_rate:float,batch_size:int,loss_type:str,train_data: List[np.ndarray],
    	train_label: List[np.ndarray],test_data: List[np.ndarray],test_label: List[np.ndarray],
    	val_data: List[np.ndarray],val_label: List[np.ndarray]):

    	loss_classes = {
    		'mse': MeanSquaredErrorLoss,
    		'ce': CrossEntropyLoss
    	}
    	if loss_type not in loss_classes:
        	raise ValueError(f"Unsupported loss type: {loss_type}")

    	loss = loss_classes[loss_type]()

    	optimizer_classes = {
    		'sgd': SGD,
    		'momentum':Momentum
    	}
    	if optimizer not in optimizer_classes:
    		raise ValueError(f"Unsupported optimizer: {optimizer}")

    	opti = optimizer_classes[optimizer](learning_rate)
    	opti.config(self.W,self.B)

    	num_batches = train_data.shape[0]//batch_size
    	for ep in range(epochs):
            for b_id in range(num_batches+1):                
                batch_start = b_id * batch_size
                batch_end = min((b_id + 1) * batch_size, train_data.shape[0])
                batch_data = train_data[batch_start:batch_end]
                batch_label = train_label[batch_start:batch_end]                

                A,H,y_prob = self.feed_forward(batch_data,self.W,self.B)
                
                dw,db = self.gradients(A,H,self.W,self.B,batch_data,y_prob,batch_label)
                opti.update(self.W,self.B,dw,db)

                
                if(b_id+1)%100 == 0:
                    _,_,test_y_prob = self.feed_forward(test_data,self.W,self.B)
                    test_loss = loss.compute(test_label,test_y_prob)
                    train_acc = self.accuracy(batch_data,batch_label)
                    train_loss = loss.compute(batch_label,y_prob)
                    test_acc = self.accuracy(test_data,test_label)                    
                    sys.stdout.write(f"\rEpoch {ep + 1}/{epochs} - Batch {b_id + 1}/{num_batches} - train-loss: {train_loss:.6f} train-acc: {train_acc:.6f} test-loss:{test_loss:.6f} test-acc : {test_acc:.6f} ")
                    sys.stdout.flush()
            print()


if __name__ == '__main__':
	(train_data,train_label),(test_data,test_label),(val_data,val_label) = Dataset('fashion_mnist').load_data()
	nn = Neural_Net(784,4,[32,32,32,32],'ReLU',10,'Xavier',0.005)
	nn.train('momentum',10,0.001,32,'ce',train_data,train_label,test_data,test_label,val_data,val_label)