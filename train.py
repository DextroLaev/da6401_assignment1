from neural_net import *
import argparse
from dataset import *


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train a neural network with configurable hyperparameters.")

    parser.add_argument("-wp", "--wandb_project", type=str, required=True,help="Project name used to track experiments in Weights & Biases dashboard.",default='cs24s031-dl-assignment')

    parser.add_argument("-we", "--wandb_entity", type=str, required=True,help="Wandb Entity used to track experiments in the Weights & Biases dashboard.",default='rajshekharrakshit')
    parser.add_argument("-d", "--dataset", type=str,choices=["mnist", "fashion_mnist"], default="fashion_mnist",help="Dataset to be used. Choices: ['mnist', 'fashion_mnist'].")
    parser.add_argument("-e", "--epochs", type=int, default=10,help="Number of epochs to train the neural network.")
    parser.add_argument("-b", "--batch_size", type=int, default=64,help="Batch size used to train the neural network.")
    parser.add_argument("-l", "--loss", type=str,default='cross_entropy', choices=["mean_squared_error", "cross_entropy"],help="Loss function to be used. Choices: ['mean_squared_error', 'cross_entropy'].")

    parser.add_argument("-o", "--optimizer", type=str,default='nadam' ,choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],help="Optimizer to be used. Choices: ['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'].")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001,help="Learning rate used to optimize model parameters.")
    parser.add_argument("-m", "--momentum", type=float, default=0.9,help="Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type=float, default=0.9,help="Beta used by the RMSprop optimizer.")

    parser.add_argument("-beta1", "--beta1", type=float, default=0.9,help="Beta1 used by Adam and Nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999,help="Beta2 used by Adam and Nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8,help="Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0,help="Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type=str, default='Xavier', choices=["random", "Xavier"], help="Weight initialization method. Choices: ['random', 'Xavier'].")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3,help="Number of hidden layers used in the feedforward neural network including output layer.")
    parser.add_argument("-sz", "--hidden_size", type=int, default=32,help="List of numbers specifying the number of hidden neurons in each layer including output layer.")
    parser.add_argument("-a", "--activation", type=str,default='ReLU', choices=["identity","sigmoid", "tanh", "ReLU"], help="Activation function to be used. Choices: ['identity', 'sigmoid', 'tanh', 'ReLU'].")
    parser.add_argument("--output_shape",type=int,default=10,help="Number of neuron in output layer.")    

    args = parser.parse_args()
    dataset = args.dataset
    
    loss_func = args.loss
    
    weight_decay = args.weight_decay
    weight_init = args.weight_init
    num_hidden_layers = args.num_layers
    num_neurons_each_layer = [ args.hidden_size ] * num_hidden_layers
    activation = args.activation
    optimizer_kwargs = {}
    if args.optimizer in ["momentum", "nag"]:
        optimizer_kwargs["momentum"] = args.momentum
    elif args.optimizer == "rmsprop":
        optimizer_kwargs["beta"] = args.beta
        optimizer_kwargs["eps"] = args.epsilon
    elif args.optimizer in ["adam", "nadam"]:
        optimizer_kwargs["beta1"] = args.beta1
        optimizer_kwargs["beta2"] = args.beta2
        optimizer_kwargs["eps"] = args.epsilon


    (train_data,train_label),(test_data,test_label),(val_data,val_label) = Dataset(dataset).load_data()
    input_shape = train_data.shape[1]
    output_shape = args.output_shape
    wandb.init(project=args.wandb_project,name=args.wandb_entity)
    
    nn = Neural_Net(input_shape = input_shape,output_shape = output_shape,
    	number_of_hidden_layers=num_hidden_layers, hidden_neurons_per_layer=num_neurons_each_layer,
    	activation_name=activation,type_of_init=weight_init,L2reg_const=weight_decay)
    nn.train(
        optimizer=args.optimizer,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        loss_type=args.loss,
        train_data=train_data,
        train_label=train_label,
        val_data=val_data,
        val_label=val_label,
        test_data = test_data,
        test_label = test_label,
        batch_size=args.batch_size,
        **optimizer_kwargs
    )
    test_loss, test_acc = nn.test_accuracy_loss(test_data,test_label)
    print('Test loss:- ',test_loss)
    print('Test Acc:- ',test_acc)
    wandb.finish()


