import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # graphs

data_train = pd.read_csv('/kaggle/input/mnist-data/train.csv')

data_train = np.array(data_train)
m, n = data_train.shape # m = number of training examples, n = number of features per example
np.random.shuffle(data_train) # shuffles rows

dev_set = data_train[0:1000].T
Y_dev = dev_set[0]
X_orig_dev = dev_set[1:n]

train_set = data_train[1000:m].T
Y_train = train_set[0]
X_orig_train = train_set[1:n]

def one_hot(Y, num_classes):
    m = Y.shape[0]
    one_hot_Y = np.zeros((num_classes, m))
    one_hot_Y[Y, np.arange(m)] = 1
    return one_hot_Y

# If Y_train is (m,) or (m,1) integers from 0-9:
Y_train = one_hot(Y_train.flatten(), 10)
Y_dev   = one_hot(Y_dev.flatten(), 10)

# flatten data
X_train = X_orig_train / 255  
X_dev = X_orig_dev / 255

layers_dims = [784, 20, 15, 15, 10]

# Using 'He Intialization'
def initialize_parameters(layers_dims):
    parameters = {}
    
    for l in range(1, len(layers_dims)):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / (layers_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))


    return parameters;

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(1, L+1):
        v["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        v["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)
        s["dW" + str(l)] = np.zeros(parameters["W" + str(l)].shape)
        s["db" + str(l)] = np.zeros(parameters["b" + str(l)].shape)

    return v, s


# Activation functions

# Used for layers 1 to L-1
def relu(Z):
    return np.maximum(0, Z)

# Used for layer L (output layer)
def softmax(Z):
    numerator = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    denominator = np.sum(Z, axis = 0, keepdims=True)
    return numerator / denominator

def relu_derivative(Z):
    return (Z > 0).astype(float)


def forward_prop(X_train, parameters):

    caches = {}
    A = X_train
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        
        Z = np.dot(W, A) + b

        if (l != L):
            A = relu(Z)
        else:
            A = softmax(Z)

        caches["A" + str(l)] = A
        caches["Z" + str(l)] = Z

    return A, caches


# Calculate cross-entropy loss
def compute_cost(A, Y_train):
    m = Y_train.shape[1]
    epsilon = 1e-8

    A_clipped = np.clip(A, epsilon, 1-epsilon)  # Make sure all values are in between (epsilon, 1-epsilon)
      
    cost = -(1/m) * np.sum(Y_train * np.log(A_clipped))
    return cost


def backward_prop(X_train, Y_train, parameters, caches):

    grads = {}
    m = Y_train.shape[0]
    L = len(parameters) // 2

    A_final = caches["A" + str(L)]
    dZ = A_final - Y_train

    A_prev = caches["A" + str(L-1)]
    grads["dW" + str(L)] = (1/m) * np.dot(dZ, A_prev.T)
    grads["db" + str(L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

    for l in reversed(range(1, L)):
        dA = np.dot(parameters["W" + str(l+1)].T, dZ)
        Z = caches["Z" + str(l)]
        dZ = dA * relu_derivative(Z)
        
        A_prev = X_train if l == 1 else caches["A" + str(l-1)]
        grads["dW" + str(l)] = (1/m) * np.dot(dZ, A_prev.T)
        grads["db" + str(l)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)


    return grads


def update_params_adam(grads, parameters, v, s, learning_rate, beta1, beta2, epsilon, t):
    L = len(parameters) // 2

    s_corrected = {}
    v_corrected = {}
    
    for l in range(1, L+1):
        # Update biased first moment estimate
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        # Update biased second raw moment estimate
        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)

        # Compute bias-corrected first moment
        v_corrected_dW = v["dW" + str(l)] / (1 - beta1 ** t)
        v_corrected_db = v["db" + str(l)] / (1 - beta1 ** t)

        # Compute bias-corrected second raw moment
        s_corrected_dW = s["dW" + str(l)] / (1 - beta2 ** t)
        s_corrected_db = s["db" + str(l)] / (1 - beta2 ** t)

        # Update parameters
        parameters["W" + str(l)] -= learning_rate * (v_corrected_dW / (np.sqrt(s_corrected_dW) + epsilon))
        parameters["b" + str(l)] -= learning_rate * (v_corrected_db / (np.sqrt(s_corrected_db) + epsilon))
    
    return parameters, v, s


def update_params(grads, parameters, learning_rate):
    L = len(parameters) // 2
    
    for l in range(1, L+1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters


def nn_model(X_train, Y_train, learning_rate, layers_dims, epochs):

    for t in range(epochs):
        # initialize parameters
    
        parameters = initialize_parameters(layers_dims)
        v, s = initialize_adam(parameters)
    
        # forward propagation
    
        A, caches = forward_prop(X_train, parameters)
    
        # compute cost
    
        cost = compute_cost(A, Y_train)
    
        # back_propagation
    
        grads = backward_prop(X_train, Y_train, parameters, caches)
    
        # update parameters
    
        parameters, v, s = update_params_adam(grads, parameters, v, s, 0.001, 0.9, 0.999, 1e-8, t)

        if (t % 100 == 0):
            print(f"Iteration {t}, cost = {cost}")

    return parameters


def predict(X, Y, parameters):
    # Forward pass
    _, AL = forward_prop(X, parameters)  # AL shape: (10, m)
    
    # Predictions = class with highest probability
    predictions = np.argmax(AL, axis=0)  # shape: (m,)
    
    # Convert Y from one-hot to label indices if needed
    if Y.ndim > 1:
        Y_labels = np.argmax(Y, axis=0)
    else:
        Y_labels = Y.flatten()
    
    # Accuracy
    accuracy = np.mean(predictions == Y_labels) * 100
    
    return predictions, accuracy

parameters = nn_model(X_dev, Y_dev, 0.001, layers_dims, 5000)

predictions, accuracy = predict(X_dev, Y_dev, parameters)
print(accuracy)


def plot_actual_vs_predicted(X, y_true, y_pred, index=0):
    # Convert one-hot to label if needed
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    plt.imshow(X[:, index].reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {y_true[index]}  Pred: {y_pred[index]}")
    plt.axis('off')
    plt.show()

plot_actual_vs_predicted(X_dev, Y_dev, predictions, 3)


