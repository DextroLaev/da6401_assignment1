import numpy as np
from loss import *
from dataset import *
from optimizer import *
from typing import List,Tuple
import sys
from activations import *
import wandb
import copy

class Neural_Net:

    def __init__(self,input_shape,number_of_hidden_layers,hidden_neurons_per_layer,activation_name,output_shape,type_of_init,L2reg_const) -> None:
        self.input_shape = input_shape
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_neurons_per_layer = hidden_neurons_per_layer
        self.output_shape = output_shape
        self.activation = self.select_activation(activation_name)
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
            activation_out = self.activation.out(linear_out)
            activation_out = np.clip(activation_out, -50, 50)            
            A.append(linear_out)
            H.append(activation_out)
            input_d = activation_out
    	y_pred = Softmax().out(np.dot(H[-1],W[-1]) + B[-1])
    	return A,H,y_pred

    def select_activation(self,name:str):
        activation_func = {
            'sigmoid': Sigmoid,
            'tanh': Tanh,
            'ReLU': ReLU,
            'softmax': Softmax,
            'identity':Linear
        }
        return activation_func[name]()

    def accuracy(self,data: List[np.ndarray],label:List[np.ndarray]) -> float:
        _,_,y_prob = self.feed_forward(data,self.W,self.B)
        predicted_labels = np.argmax(y_prob,axis=1)
        true_labels = np.argmax(label,axis=1)
        acc = np.mean(predicted_labels==true_labels)
        return acc

    def gradients(self,A: List[np.ndarray],H:List[np.ndarray],W:List[np.ndarray],B:List[np.ndarray],batch_data: List[np.ndarray], y_prob: List[np.ndarray], batch_label: List[np.ndarray]):
        dW = []
        db = []
        if isinstance(self.loss,MeanSquaredErrorLoss):
            dl_dh = y_prob - batch_label
            ans = []
            for i, row in enumerate(y_prob):
                dh_da = np.diag(row) - np.dot(row.T, row)
                ans += [np.dot(dl_dh[i], dh_da)]
            error = np.array(ans)
            
        else:
            error = y_prob - batch_label

    	# Backpropagate through output layer (softmax)
        dW.insert(0, np.dot(H[-1].T, error)+self.L2reg_const * W[-1])
        db.insert(0, np.sum(error,axis=0,keepdims=True))

    	# Backpropagate through hidden layers
        delta = error
        for i in range(self.number_of_hidden_layers - 1, 0, -1):
            delta = np.dot(delta, W[i+1].T) * self.activation.derivative(A[i])
            dW.insert(0, np.dot(H[i-1].T, delta)+ self.L2reg_const * W[i])
            db.insert(0, np.sum(delta, axis=0,keepdims=True))

    	# Backpropagate through input layer
        delta = np.dot(delta, W[1].T) * self.activation.derivative(A[0])
        dW.insert(0, np.dot(batch_data.T, delta)+ self.L2reg_const * W[0])
        db.insert(0, np.sum(delta, axis=0,keepdims=True))
        # print(dW)
        # input()
        return dW, db    
    
    def get_weights(self,opti,W,B):
        if isinstance(opti,Nestrov):
            W = [w - opti.beta * mw for w, mw in zip(self.W, opti.optimizer_config['momentum_W'])]
            B = [b - opti.beta * bw for b, bw in zip(self.B, opti.optimizer_config['momentum_B'])]
        return W,B

    def train(self,optimizer:str,epochs:int,learning_rate:float,batch_size:int,loss_type:str,train_data: List[np.ndarray],
    	train_label: List[np.ndarray],test_data: List[np.ndarray],test_label: List[np.ndarray],
    	val_data: List[np.ndarray],val_label: List[np.ndarray],**optimizer_kwargs):
        
        loss_classes = {
            'mean_squared_error': MeanSquaredErrorLoss,
            'cross_entropy': CrossEntropyLoss
        }
        if loss_type not in loss_classes:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        self.loss = loss_classes[loss_type]()

        optimizer_classes = {
            'sgd': SGD,
            'momentum':Momentum,
            'rmsProp':RMSProp,
            'adam':Adam,
            'nadam':Nadam,
            'nag':Nestrov
            }
        if optimizer not in optimizer_classes:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        opti = optimizer_classes[optimizer](learning_rate,**optimizer_kwargs)        
        opti.config(self.W,self.B)
        num_batches = train_data.shape[0]//batch_size
        for ep in range(epochs):
            for b_id in range(num_batches+1):
                
                batch_start = b_id * batch_size
                batch_end = min((b_id + 1) * batch_size, train_data.shape[0])
                batch_data = train_data[batch_start:batch_end]
                batch_label = train_label[batch_start:batch_end]

                W,B = self.get_weights(opti,copy.deepcopy(self.W),copy.deepcopy(self.B))
                A, H, y_prob = self.feed_forward(batch_data, W, B)
                
                dw,db = self.gradients(A,H,W,B,batch_data,y_prob,batch_label)
                opti.update(self.W,self.B,dw,db)

                
                if(b_id+1)%100 == 0:
                    _,_,val_y_prob = self.feed_forward(val_data,self.W,self.B)
                    val_loss = self.loss.compute(val_y_prob,val_label)
                    val_acc = self.accuracy(val_data,val_label)
                    
                    train_acc = self.accuracy(batch_data,batch_label)
                    train_loss = self.loss.compute(y_prob,batch_label)
                    
                    wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc}) 
                    sys.stdout.write(f"\rEpoch {ep + 1}/{epochs} - Batch {b_id + 1}/{num_batches} - train-loss: {train_loss:.6f} train-acc: {train_acc:.6f} val-loss:{val_loss:.6f} val-acc : {val_acc:.6f} ")
                    sys.stdout.flush()
            print()

    def test_accuracy_loss(self,data,label):
        _,_,y_prob = self.feed_forward(data,self.W,self.B)
        loss_val = self.loss.compute(y_prob,label)
        acc = self.accuracy(data,label)
        return loss_val, acc

if __name__ == '__main__':
	(train_data,train_label),(test_data,test_label),(val_data,val_label) = Dataset('fashion_mnist').load_data()
	nn = Neural_Net(784,4,[32,32,32,32],'ReLU',10,'Xavier',0.005)
	nn.train('Adam',10,0.0001,32,'mse',train_data,train_label,test_data,test_label,val_data,val_label)