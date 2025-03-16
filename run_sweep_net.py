from neural_net import *
import wandb
from dataset import *

def train():
    var1 = wandb.init(project='dl-assignment1')
    config = var1.config

    (train_data,train_label),(test_data,test_label),(val_data,val_label) = Dataset('fashion_mnist').load_data()
    input_size = train_data.shape[1]
    num_hidden_layers = config.num_of_hidden_layers

    num_neurons_each_layer = [ config.hidden_layer_size ] * num_hidden_layers 
    activation_function = config.activation_function
    type_of_init = config.weight_initialization
    L2reg_const = config.weight_decay

    run_name = f"hl_{config.num_of_hidden_layers}_bs_{config.batch_size}_ac_{config.activation_function}"
    
    wandb.run.name = run_name
    wandb.run.save()
    optimizer_kwargs = {
        'beta':0.9,
        'beta1':0.9,
        'beta2':0.999,
        'eps':1e-8,
        'momentum':0.9
    }

    print(f"Starting training with run name: {run_name}")
    nn = Neural_Net(
        number_of_hidden_layers=num_hidden_layers,
        hidden_neurons_per_layer=num_neurons_each_layer,
        activation_name=activation_function,
        input_shape=input_size,
        type_of_init=type_of_init,
        L2reg_const=L2reg_const,
        output_shape = 10,
    )
    
    nn.train(
        optimizer=config.optimizer,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        train_data=train_data,  # Make sure to define train_data and labels
        train_label=train_label,
        test_data=test_data,
        test_label=test_label,
        val_data = val_data,
        val_label = val_label,
        loss_type = 'cross_entropy',
        batch_size=config.batch_size,
        **optimizer_kwargs
    )

if __name__ == '__main__':
    sweep_config = {
		'name': 'fashion-mnist-exp(bayes-select)-3.3.1-ce exp',
		'method': 'bayes',
		'metric': {'goal': 'maximize', 'name': 'val_acc'},
		'parameters': {
		    'num_of_hidden_layers':{'values':[3,4,5]},
		    'hidden_layer_size': {'values': [32, 64, 128]},
		    'activation_function': {'values': ['sigmoid', 'tanh', 'ReLU']},
		    'batch_size': {'values': [16, 32, 64]},
		    'epochs': {'values': [5, 10]},
		    'learning_rate': {'values': [1e-3,1e-4]},
		    'optimizer': {'values': ['sgd', 'momentum', 'nag', 'rmsProp', 'adam', 'nadam']},
		    'weight_initialization': {'values': ['random', 'Xavier']},
		    'weight_decay': {'values': [0, 0.0005,0.5]},
		  }
    }
    sweep_id = wandb.sweep(sweep_config,project='dl-assignment1')
    wandb.agent(sweep_id,train,count=50)
    wandb.finish()