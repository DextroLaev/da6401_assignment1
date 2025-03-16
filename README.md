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

-**`train.py`**
  - This file is used to train and evaluate the model.
  - This file takes the following commnad-line argument to run the neural network:
        - wandb_project : Name of your project.
        - wandb_entity : Name of the run in Wandb.
        - dataset: Name of the dataset. Currently supports **`mnist`** and **`fashion_mnist`**.
        - epochs: Number of epochs you want to train our model for.
        - batch_size: size of each batch
        - loss: which loss to use **`mean_square_error`** or **`cross_entropy`**
        - learning_rate: What learaning rate to use
        
    
## 🚀 Training the Neural Network

To train the neural network, run the following command:

```bash
python train.py --wandb_project <project_name> --wandb_entity <entity_name> --dataset <dataset> --epochs <num_epochs> --batch_size <batch_size> --loss <loss_function> --learning_rate <lr> --momentum <momentum> --beta <beta> --beta1 <beta1> --beta2 <beta2> --epsilon <epsilon> --weight_decay <weight_decay> --weight_init <weight_init> --num_layers <num_layers> --hidden_size <hidden_size> --activation <activation> --output_shape <output_shape>
