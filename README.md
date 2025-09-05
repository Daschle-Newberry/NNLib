# NNLib

NNLib is a lightweight python library used for building and training small neural networks.


## About

---
NNLib is a passion project of mine, and was primarily an educational exercise to learn more about training and building neural networks.
The project is completely standalone besides its use of NumPy for numerical computation and MatPlotLib for data visualization.


## How to Build

---
1. Clone the repo by using the following command in the terminal
```commandline
git clone https://github.com/Daschle-Newberry/NNLib
```
2. Install the dependencies

```commandline
pip install numpy matplotlib
```

3. **Optional:** if you would like to run the MNIST example, you will need to install pygame
```commandline
pip install pygame
```

## Example
```python

#Define XOR data set
X = np.array([
        [0,0],
        [0,1],
        [1,0],
        [1,1]
    ])
    y = np.array([
        0,
        1,
        1,
        0
    ])
    
    #Reshape y to be a row vector to match input training data
    y = y.reshape(-1,1)
    
    #Optionally tile data for more gradient updates per epoch
    X = np.tile(X,(10000,1))
    y = np.tile(y,(10000,1))
    
    
    #Define model
    model = Sequential(
        [
            Dense(3,"Dense1"),
            Sigmoid(),
            Dense(1,"Dense2"),
            Sigmoid(),
        ], BinaryCrossEntropy()
    )
    
    
    #Compile the model
    model.compile(input_size=(2,), optimizer=Adam(learning_rate=.01, beta1=.9, beta2=.99))
    
    
    #Train model
    model.fit(X, y, epochs = 5, batch_size = 32)
    
    
    #Evaluate predictions
    prediction = (model.predict(X) > .5)
    model_accuracy = np.mean(prediction == y)

    print(f"Model accuracy is {model_accuracy}")
```

## Features

---

NNLib supports a variety of activations, optimizers, layer types, and loss functions. All of these features were implemented from scratch, and are all numerically stable, fully **batch** vectorized in NumPy, and ready for GPU acceleration. 

### Layer Types 

- Dense (fully connected)
- Convolution2D
- MaxPool2D
- Flatten (Reshape)

### Activation Functions

- ReLu
- Softmax (integrated with cross-entropy loss)
- Sigmoid
- TanH

### Loss Functions

- Binary Cross Entropy
- Mean Squared Error
- Categorical Cross Entropy (with built-in softmax)

### Optimizers

- Stochastic Gradient Descent (SGD)
- Stochastic Gradient Descent with Momentum (Momentum)
- RMSProp
- Adam


