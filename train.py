import numpy as np
import wandb
import sys
import math

from keras.datasets import fashion_mnist,mnist
import sklearn
from sklearn.model_selection import train_test_split
import argparse

def load_dataset(dataset='fashion_mnist'):
	if dataset == 'fashion_mnist':
		(train_data,train_label),(test_data,test_label) = fashion_mnist.load_data()
	elif dataset == 'mnist':
		(train_data,train_label),(test_data,test_label) = fashion_mnist.load_data()

	return (train_data,train_label),(test_data,test_label)

def format_data(train_data,train_label,test_data,test_label):
	train_data, train_label = sklearn.utils.shuffle(train_data, train_label)
	test_data, test_label = sklearn.utils.shuffle(test_data, test_label)

	train_data = train_data.reshape(train_data.shape[0],-1)
	test_data = test_data.reshape(test_data.shape[0],-1)
	train_data = train_data/255.0
	test_data = test_data/255.0
	train_label = np.eye(10)[train_label]
	test_label = np.eye(10)[test_label]

	train_data,val_data,train_label,val_label = train_test_split(train_data,train_label,test_size=0.1,random_state=42)
	return train_data,train_label,test_data,test_label,val_data,val_label


class Neural_net:
    def __init__(self,num_hidden_layers,num_neurons_each_layer,activation_function,input_size,type_of_init,L2reg_const=0):
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_each_layer = num_neurons_each_layer
        self.input_size = input_size
        self.W,self.B = self.init_parameters(type_of_init)
        self.activation_function = activation_function
        self.L2reg_const = L2reg_const
        

    def init_parameters(self,init):
        W = []
        B = []
        if init == 'xavier':
            W.append(np.random.randn(self.num_neurons_each_layer[0], self.input_size) * np.sqrt(1 / self.input_size))
            for i in range(1,self.num_hidden_layers):
                W.append(np.random.randn(self.num_neurons_each_layer[i], self.num_neurons_each_layer[i-1]) * np.sqrt(1 / self.num_neurons_each_layer[i-1]))
        else:
            W.append(np.random.randn(self.num_neurons_each_layer[0], self.input_size) * 0.01)
            for i in range(1, self.num_hidden_layers):
                W.append(np.random.randn(self.num_neurons_each_layer[i], self.num_neurons_each_layer[i - 1]) * 0.01)

        B.append(np.zeros(shape=(1,self.num_neurons_each_layer[0])))
        for i in range(1,self.num_hidden_layers):
            B.append(np.zeros(shape=(1,self.num_neurons_each_layer[i])))

        return W, B

    def activation(self,x):
        if self.activation_function == 'ReLU':
            return self.ReLU(x)
        elif self.activation_function == 'tanh':
            return self.tanh(x)
        elif self.activation_function == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation_function == 'identity':
            return x

    def activation_derivative(self,x):
        if self.activation_function == 'ReLU':
            return self.ReLU_derivative(x)
        elif self.activation_function == 'tanh':
            return self.tanh_derivative(x)
        elif self.activation_function == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation_function == 'identity':
            return 1

    def tanh(self,x):
        return np.array([((np.exp(z) - np.exp(-z))/((np.exp(z) + np.exp(-z)))) for z in x])

    def tanh_derivative(self,x):
        return np.array(1 - self.tanh(x)**2)
    
    def softmax_activation(self,x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def ReLU(self,x):
        return np.maximum(0,x)

    def ReLU_derivative(self,x):
        return np.where(x > 0, 1, 0)
    
    def sigmoid(self,x):
        x = np.clip(x, -500, 500)
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def FeedForward(self,data):
        a_l = []
        h_l = []
        input_linear_out = np.dot(data,self.W[0].T) + self.B[0]
        a_l.append(input_linear_out)
        for i in range(1,self.num_hidden_layers):
            activation_out = self.activation(a_l[-1])
            h_l.append(activation_out)
            linear_out = np.dot(h_l[-1],self.W[i].T) + self.B[i]
            a_l.append(linear_out)


        y_hat = self.softmax_activation(a_l[-1])
        return a_l,h_l,y_hat

    def val_loss_and_acc(self,val_data,val_label):
        acc = 0
        error = 0
        val_loss = []
        val_acc = []
        
        for d in range(val_data.shape[0]):
            a_li,h_li,y_hat_i = self.FeedForward(val_data[d].reshape(1,-1))
            s = [x.sum() for x in self.W]
            error += -np.sum(val_label[d] * np.log(y_hat_i[0])) + self.L2reg_const/2*sum(s)
            if np.argmax(val_label[d]) == np.argmax(y_hat_i[0]):
              acc += 1
        return error/val_data.shape[0],acc/val_data.shape[0]

    def backpropagation(self,A,H,y_hat,y,data):
      dA = []
      dH = []
      dW = []
      dB = []
      dA.append(-(y - y_hat))
      for i in range(self.num_hidden_layers-1,-1,-1):
        if i == self.num_hidden_layers - 1:
            grad_val_w = np.dot(dA[-1].reshape(-1, 1), H[-1].reshape(1, -1))
        elif i == 0:
          grad_val_w = np.dot(dA[-1].reshape(-1,1),data.reshape(1,-1))
        else:
          grad_val_w = np.dot(dA[-1].reshape(-1,1),H[i-1].reshape(1,-1))
        dW.append(grad_val_w)
        grad_val_b = np.sum(dA[-1],axis=0,keepdims=True)
        dB.append(grad_val_b)

        if i > 0:
          grad_L_hi = np.dot(self.W[i].T,dA[-1].T)
          grad_L_ai = np.multiply(grad_L_hi.T,self.activation_derivative(A[i-1]))
          dA.append(grad_L_ai)
      return dW[::-1],dB[::-1]

    def train(self,optimizer,epochs,learning_rate,data,label,val_data,val_label,batch_size=32,**kwargs):
        num_batches = len(data)//batch_size
        optimizer_config = {'learning_rate':learning_rate,**kwargs}
        if optimizer == 'momentum':
            momentum_W = [np.zeros_like(w) for w in self.W]
            momentum_B = [np.zeros_like(b) for b in self.B]
            optimizer_config['momentum_W'] = momentum_W
            optimizer_config['momentum_B'] = momentum_B
        elif optimizer == 'nesterov':
            momentum_W = [np.zeros_like(w) for w in self.W]
            momentum_B = [np.zeros_like(b) for b in self.B]
            optimizer_config['momentum_W'] = momentum_W
            optimizer_config['momentum_B'] = momentum_B
        elif optimizer == 'RMSprop':
            v_W = [np.zeros_like(w) for w in self.W]
            v_B = [np.zeros_like(b) for b in self.B]
            optimizer_config['v_W'] = v_W
            optimizer_config['v_B'] = v_B
        elif optimizer == 'adam' or optimizer == "nadam":
            momentum1_W = [np.zeros_like(w) for w in self.W]
            momentum1_B = [np.zeros_like(b) for b in self.B]
            momentum2_W = [np.zeros_like(w) for w in self.W]
            momentum2_B = [np.zeros_like(b) for b in self.B]
            optimizer_config['momentum1_W'] = momentum1_W
            optimizer_config['momentum1_B'] = momentum1_B
            optimizer_config['momentum2_W'] = momentum2_W
            optimizer_config['momentum2_B'] = momentum2_B
            optimizer_config['t'] = 0

        for ep in range(epochs):
            train_loss = 0
            train_acc = 0
            for batch in range(num_batches):
                start = batch*batch_size
                end = (batch+1)*batch_size
                batch_data = data[start:end]
                batch_label = label[start:end]
                batch_dw = [np.zeros_like(w) for w in self.W]
                batch_db = [np.zeros_like(b) for b in self.B]
                batch_correct = 0
                if optimizer in ('adam','nadam'):
                    optimizer_config['t'] += 1

                for i in range(batch_size):
                    a_li,h_li,y_hat_i = self.FeedForward(batch_data[i].reshape(1,-1))
                    s = [x.sum() for x in self.W]
                    train_loss += -np.sum(batch_label[i] * np.log(y_hat_i[0])) + self.L2reg_const/2*sum(s)
                    # train_loss += -np.sum(batch_label[i].reshape(1, -1) * np.log(y_hat_i[0].reshape(1, -1))) + self.L2reg_const/2*sum(s)

                    if np.argmax(batch_label[i]) == np.argmax(y_hat_i[0]):
                        batch_correct += 1
                    dw, db = self.backpropagation(a_li, h_li, y_hat_i[0], batch_label[i], batch_data[i])
                    for k in range(self.num_hidden_layers):
                        batch_dw[k] += dw[k]
                        batch_db[k] += db[k]
                train_acc += batch_correct
                self.W,self.B,optimizer_config = self.apply_optimizer(optimizer,batch_dw,batch_db,optimizer_config,batch_size)
                if (batch + 1) % 10 == 0:
                    val_loss, val_acc = self.val_loss_and_acc(val_data, val_label)
                    # wandb.log({'epoch': ep + 1, 'train_loss': train_loss / ((batch+1)*batch_size), 'train_acc': train_acc / ((batch+1)*batch_size), 'val_loss': val_loss, 'val_acc': val_acc})
                    sys.stdout.write(f"\rEpoch {ep + 1}/{epochs} - Batch {batch + 1}/{num_batches} - Loss: {train_loss / ((batch+1)*batch_size):.6f} Train-Acc: {train_acc / ((batch+1)*batch_size):.6f} val-loss:{val_loss:.6f} val-Acc: {val_acc:.6f} ")
                    sys.stdout.flush()
            print()
    def apply_optimizer(self,optimizer,batch_dw,batch_db,config,batch_size):
        learning_rate = config['learning_rate']
        if optimizer == 'sgd':
            for k in range(self.num_hidden_layers):
                self.W[k] -= (learning_rate/batch_size)*batch_dw[k]
                self.B[k] -= (learning_rate/batch_size)*batch_db[k]
        elif optimizer == 'momentum':
            momentum_W = config['momentum_W']
            momentum_B = config['momentum_B']
            beta = config.get('beta',0.6)
            for k in range(self.num_hidden_layers):
                momentum_W[k] = beta*momentum_W[k] + batch_dw[k]
                momentum_B[k] = beta*momentum_B[k] + batch_db[k]
                self.W[k] -= learning_rate*momentum_W[k]
                self.B[k] -= learning_rate*momentum_B[k]
            config['momentum_W'] = momentum_W
            config['momentum_B'] = momentum_B
        elif optimizer == 'nesterov':
            momentum_W = config['momentum_W']
            momentum_B = config['momentum_B']
            beta = config.get('beta',0.6)
            for k in range(self.num_hidden_layers):
                W_lookahead = [w - beta * v for w, v in zip(self.W, momentum_W)]
                B_lookahead = [b - beta * v for b, v in zip(self.B, momentum_B)]
                
                momentum_W[k] = beta * momentum_W[k] + (learning_rate / batch_size) * batch_dw[k]
                momentum_B[k] = beta * momentum_B[k] + (learning_rate / batch_size) * batch_db[k]
                self.W[k] = W_lookahead[k] - momentum_W[k]
                self.B[k] = B_lookahead[k] - momentum_B[k]

            config['momentum_W'] = momentum_W
            config['momentum_B'] = momentum_B

        elif optimizer == 'RMSprop':
            v_W = config['v_W']
            v_B = config['v_B']
            beta = config.get('beta', 0.9)
            eps = config.get('eps', 1e-8)
            for k in range(self.num_hidden_layers):
                v_W[k] = beta * v_W[k] + (1 - beta) * (batch_dw[k]**2)
                v_B[k] = beta * v_B[k] + (1 - beta) * (batch_db[k]**2)

                adaptive_lr_w = (learning_rate / (np.sqrt(v_W[k]) + eps))
                adaptive_lr_b = (learning_rate / (np.sqrt(v_B[k]) + eps))
                self.W[k] -= adaptive_lr_w * batch_dw[k]
                self.B[k] -= adaptive_lr_b * batch_db[k]

            config['v_W'] = v_W
            config['v_B'] = v_B
        elif optimizer == 'adam':
            momentum1_W = config['momentum1_W']
            momentum1_B = config['momentum1_B']
            momentum2_W = config['momentum2_W']
            momentum2_B = config['momentum2_B']
            t = config['t']
            beta1 = config.get('beta1', 0.9)
            beta2 = config.get('beta2', 0.999)
            eps = config.get('eps', 1e-8)

            for i in range(self.num_hidden_layers):
                momentum1_W[i] = beta1*momentum1_W[i] + (1-beta1)*(batch_dw[i])
                momentum1_B[i] = beta1*momentum1_B[i] + (1-beta1)*(batch_db[i])

                momentum2_W[i] = beta2*momentum2_W[i] + (1-beta2)*(batch_dw[i]**2)
                momentum2_B[i] = beta2*momentum2_B[i] + (1-beta2)*(batch_db[i]**2)

                momentum1_W_hat = momentum1_W[i]/(1-(beta1**t))
                momentum1_B_hat = momentum1_B[i]/(1-(beta1**t))

                momentum2_W_hat = momentum2_W[i]/(1-(beta2**t))
                momentum2_B_hat = momentum2_B[i]/(1-(beta2**t))

                adaptive_lr_W = learning_rate/(np.sqrt(momentum2_W_hat) + eps)
                adaptive_lr_B = learning_rate/(np.sqrt(momentum2_B_hat) + eps)

                self.W[i] -= adaptive_lr_W * momentum1_W_hat
                self.B[i] -= adaptive_lr_B * momentum1_B_hat
            config['momentum1_W'] = momentum1_W
            config['momentum1_B'] = momentum1_B
            config['momentum2_W'] = momentum2_W
            config['momentum2_B'] = momentum2_B
            config['t'] = t
        elif optimizer == 'nadam':
            momentum1_W = config['momentum1_W']
            momentum1_B = config['momentum1_B']
            momentum2_W = config['momentum2_W']
            momentum2_B = config['momentum2_B']
            t = config['t']
            beta1 = config.get('beta1', 0.9)
            beta2 = config.get('beta2', 0.999)
            eps = config.get('eps', 1e-8)

            for i in range(self.num_hidden_layers):
                momentum1_W[i] = beta1 * momentum1_W[i] + (1 - beta1) * (batch_dw[i])
                momentum1_B[i] = beta1 * momentum1_B[i] + (1 - beta1) * (batch_db[i])

                momentum2_W[i] = beta2 * momentum2_W[i] + (1 - beta2) * (batch_dw[i]**2)
                momentum2_B[i] = beta2 * momentum2_B[i] + (1 - beta2) * (batch_db[i]**2)

                momentum1_W_hat = momentum1_W[i] / (1 - (beta1**t))
                momentum1_B_hat = momentum1_B[i] / (1 - (beta1**t))

                momentum2_W_hat = momentum2_W[i] / (1 - (beta2**t))
                momentum2_B_hat = momentum2_B[i] / (1 - (beta2**t))

                m_nestrov_W = beta1 * momentum1_W_hat + ((1 - beta1) * batch_dw[i])/(1-beta1**t)
                m_nestrov_B = beta1 * momentum1_B_hat + ((1 - beta1) * batch_db[i])/(1-beta1**t)

                adaptive_lr_W = learning_rate / (np.sqrt(momentum2_W_hat) + eps)
                adaptive_lr_B = learning_rate / (np.sqrt(momentum2_B_hat) + eps)

                self.W[i] -= adaptive_lr_W * m_nestrov_W
                self.B[i] -= adaptive_lr_B * m_nestrov_B
            config['momentum1_W'] = momentum1_W
            config['momentum1_B'] = momentum1_B
            config['momentum2_W'] = momentum2_W
            config['momentum2_B'] = momentum2_B
            config['t'] = t

        return self.W,self.B,config


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train a neural network with configurable hyperparameters.")
    parser.add_argument("-wp", "--wandb_project", type=str, required=True,help="Project name used to track experiments in Weights & Biases dashboard.",default='myproject')
    parser.add_argument("-we", "--wandb_entity", type=str, required=True,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='myname')
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist",help="Dataset to be used. Choices: ['mnist', 'fashion_mnist'].")
    parser.add_argument("-e", "--epochs", type=int, default=1,help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=4,help="Batch size used to train the neural network.")
    parser.add_argument("-l", "--loss", type=str, choices=["mean_squared_error", "cross_entropy"], default="cross_entropy",help="Loss function to be used. Choices: ['mean_squared_error', 'cross_entropy'].")

    parser.add_argument("-o", "--optimizer", type=str, choices=["sgd", "momentum", "nag", "RMSprop", "adam", "nadam"], default="sgd",help="Optimizer to be used. Choices: ['sgd', 'momentum', 'nag', 'RMSprop', 'adam', 'nadam'].")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1,help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m", "--momentum", type=float, default=0.5,help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.5,help="Beta used by the RMSprop optimizer.")

    parser.add_argument("-beta1", "--beta1", type=float, default=0.5,help="Beta1 used by Adam and Nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5,help="Beta2 used by Adam and Nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6,help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "Xavier"], default="random",help="Weight initialization method. Choices: ['random', 'Xavier'].")
    parser.add_argument("-nhl", "--num_layers", type=int, default=2,help="Number of hidden layers used in the feedforward neural network including output layer.")
    parser.add_argument("-sz", "--hidden_size", type=int, nargs='+', default=[16,10],help="List of numbers specifying the number of hidden neurons in each layer including output layer.")
    parser.add_argument("-a", "--activation", type=str, choices=["identity","sigmoid", "tanh", "ReLU"], default="sigmoid",help="Activation function to be used. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU'].")

    args = parser.parse_args()
    dataset = args.dataset
    
    loss_func = args.loss
    
    weight_decay = args.weight_decay
    weight_init = args.weight_init
    num_hidden_layers = args.num_layers
    num_neurons_each_layer = args.hidden_size
    activation = args.activation
    optimizer_kwargs = {}
    if args.optimizer in ["momentum", "nesterov"]:
        optimizer_kwargs["beta"] = args.momentum
    elif args.optimizer == "RMSprop":
        optimizer_kwargs["beta"] = args.beta
        optimizer_kwargs["eps"] = args.epsilon
    elif args.optimizer in ["adam", "nadam"]:
        optimizer_kwargs["beta1"] = args.beta1
        optimizer_kwargs["beta2"] = args.beta2
        optimizer_kwargs["eps"] = args.epsilon

    (train_data,train_label),(test_data,test_label) = load_dataset(dataset)
    train_data,train_label,test_data,test_label,val_data,val_label = format_data(train_data,train_label,test_data,test_label)
    nn = Neural_net(num_hidden_layers=num_hidden_layers, num_neurons_each_layer=num_neurons_each_layer,activation_function=activation,input_size=784,type_of_init=weight_init,L2reg_const=weight_decay)
    nn.train(
        optimizer=args.optimizer,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        data=train_data,
        label=train_label,
        val_data=val_data,
        val_label=val_label,
        batch_size=args.batch_size,
        **optimizer_kwargs
    )

