# 🧠 Neural Network from Scratch (Using Only NumPy)

This repository contains the code for implementing a neural network from scratch using **only NumPy**.

## 📂 Repository Structure

### 🔬 Experimentation Notebooks (`.ipynb` files)
- These Jupyter Notebook files were initially used to **test** and **evaluate** the neural network.  
- They are primarily for **experimentation** and performance checks.  
- These files are **not** part of the actual working code.

### ⚙️ Core Python Files (`.py` files)
These files contain the essential code for building and running the neural network:

- **`dataset.py`** 📊  
  - Handles **data loading** and **pre-processing**.  
  - Currently supports **Fashion-MNIST** and **MNIST** datasets.

- **`activations.py`** ⚡  
  - Implements various **activation functions** along with their derivatives.
  - Currently it supports **`sigmoid`**, **`ReLU`**, **`Tanh`**, **`identity`**
  - New activation functions can be added by **inheriting the `Activation` base class** and modifying the necessary code.

- **`loss.py`** 📊  
  - Implements different loss functions.
  - Currently supports **Mean Squared Error (MSE)** and **Cross-Entropy** loss.
  - New loss functions can be added by **inheriting the `Loss` base class** and modifying the necessary code.

- **`optimizers.py`** ⚡  
  - Implements various **optimizers** along with their update rules.  
  - Currently supports **SGD, Nesterov, Momentum, Adam, and Nadam**.
  - New optimizers can be added by **inheriting the `Optimizers` class** and modifying the `config` and `update` methods.

- **`neural_net.py`** 🤖  
  - Implements the **Neural Network architecture**.
  - Contains the `Neural_Net` class with methods like `feed_forward`, `backpropagation`, etc.
  - The code is modular, allowing easy modifications to the neural network.
  - **Note:** This file only contains the algorithm of the neural network and is not meant to be executed directly.

- **`run_sweep_net.py`** 🤖  
  - This is used to log the selected parameters in the `wandb` platform.
  - One can check the performance of the neural network on various hyperparameter by running this file and checking the `wandb` site.
  - Currently it logs `train loss`, `train Accuracy`, `validation loss` and `validation accuracy`.
  - One can change the `sweep_config` present inside the code, to check the performance of neural network on different hyperparameters.


**`train.py`**
  - This file is used to train and evaluate the model.
  - This file takes the following commnad-line argument to run the neural network:
                
    | Argument        | Description |
    |----------------|-------------|
    | `--wandb_project`  | Name of your Weights & Biases (Wandb) project. |
    | `--wandb_entity`   | Name of the run in Wandb. |
    | `--dataset`        | Dataset to use: **`mnist`** or **`fashion_mnist`**. |
    | `--epochs`        | Number of training epochs. |
    | `--batch_size`    | Size of each training batch. |
    | `--loss`          | Loss function: **`mean_square_error`** or **`cross_entropy`**. |
    | `--optimizers`    | One can specify which optimizer to use , choices = [`sgd`,`momentum`,`nag`,`rmsprop`,`adam`,`nadam`] |
    | `--learning_rate` | Learning rate for optimization. |
    | `--momentum`      | This value is need when you are using momentum based optimzers ( **`Momentum`**,**`NAG`**) |
    | `--beta1`         | This value is used by **`Adam `** and **`Nadam`** optimizers |
    | `--beta2`         | This value is used by **`Adam `** and **`Nadam`** optimizers |
    | `--epsilon`         | This is used by **`RMSprop`**, **`Adam`** and **`Nadam`** optimizers |
    | `--weight_decay`      | This the L2 regularization constant , the value ranges between 0 to 1 |
    | `--weight_init`      | Weight initialization method, currectly supports **`random`** and **`Xavier`** |
    | `--num_layers`    | Number of hidden layers to use in the neural network |
    | `--hidden_size`   | List of numbers specifying the number of hidden neurons in each layer |
    | `--activation`    | Used to select which activation function to use, currently supports **`sigmoid`**,**`tanh`**,**`ReLU`**, **`Identity`** |
    | `--output_shape`   | Number of neurons in the output layer, currently for mnist and fashion_mnist it has been set to 10 by default | 
    
    
## 🚀 Training the Neural Network

First install the necessary dependencies
```bash
pip3 install -r requirements.txt
```

To train the neural network,the general run command is the following:

```bash
python train.py --wandb_project <project_name> --wandb_entity <entity_name> --dataset <dataset_name> --epochs <num_epochs> --batch_size <batch_size> --loss <loss_function> --learning_rate <lr> --momentum <momentum> --beta <beta> --beta1 <beta1> --beta2 <beta2> --epsilon <epsilon> --weight_decay <weight_decay> --weight_init <weight_init> --num_layers <num_layers> --hidden_size <hidden_size> --activation <activation> --output_shape <output_shape>

```
For Example, To train the model with adam optimizer, ReLU acivation, with weight_decay, etc.. paste the following code in the command line:
```bash
python train.py --wandb_project project_1 --wandb_entity fashion_mnist_run --dataset fashion_mnist --epochs 10 --batch_size 32 --loss cross_entropy --learning_rate 0.0001 --beta1 0.9 --beta2 0.999 --epsilon 1e-8 --weight_decay 0.0005 --weight_init Xavier --num_layers 4 --hidden_size 64 --activation ReLU --output_shape 10
```
After the training , it will automatically give you the test loss and test accuracy in the command line / terminal.




