# NeuralScaleID
## Model class (NeuralNet.py)
Instantiate a model via:
```python
import NeuralNet as NN
model = NN.Model(architecture=[10,64,64,64,2],loss='KL',softmax=True)
```
## Methods
### train
```python
train(x,y,batch_size=32,epochs=1,learning_rate=0.001,lr_scheduler = None,verbose=2,asymp_steps=1000,skip=100)
```
trains the model.
#### Arguments
* **x:** training input, numpy array of shape ```[batch size,input dimension] ```
* **y:** training output, numpy array of shape ```[batch size,output dimension]```
* **batch_size:** Integer.
* **learning_rate:** Adam learning rate.
* **lr_scheduler:** Learning rate scheduler. This is a function that takes in epoch and returns a float value for learning rate. The learning rate is updated at each training step and not just at the end of each epoch. If ```learning_rate``` and ```lr_scheduler``` are both specified, ```lr_scheduler``` overrides ```learning_rate```.
* **verbose:** Integer. 0,1 or 2. 0 = silent, 1 = epoch number, 2 = progress % within an epoch
* **asymp_steps:** The method returns the training loss for the last ```asymp_steps``` steps of the training session (the last asymp_steps of the last epoch of the training session)
* **skip:** The method returns training loss at every ```skip``` steps. set ```skip=1``` for training loss for all training steps
### predict
```python
predict(x)
```
generates output predictios for input ```x```
#### Arguments
* **x:** input, numpy array of shape ```[batch size,input dimension]```. The entire input array is evaluated as one batch

### predict_layer
```python
predict_layer(x,layer=None)
```
returns the predicted output of a specified layer for input ```x```
#### Arguments
* **x:** input, numpy array of shape ```[batch size,input dimension]```. The entire input array is evaluated as one batch
* **layer:** Integer. 1, 2... for 1st, 2nd... hidden layer, or 0, -1,... for output layer, prefinal layer... etc. If ```None```, returns a dictionary of all layer outputs. 

### evaluate
```python
evaluate(x,y)
```
evaluates loss given input ```x``` and output ```y```
#### Arguments
* **x:** input, numpy array of shape ```[batch size,input dimension]```. The entire input array is evaluated as one batch
* **y:** output, numpy array of shape ```[batch size,output dimension]```

### get_weights
```python
get_weights()
```
returns ```W``` and ```b``` (weights and biases) as lists of numpy arrays

### save_model
```python
save_model(name,location='.')
```
saves the model
#### Arguments
* **name:** String. The name under which you want to save the model
* **location** String. The directory in which you want to save your model.

## Other methods in NeuralNet.py
### load_model
```python
load_model(name,folder='.')
```
loads the model saved via ```save_model``` in ```Model``` class
### Arguments
* **name:** String. The name of the saved model
* **folder:** String. The directory containing the model

## Teacher class (teacher.py)
Instantiate a teacher via:
```python
import teacher
t = teacher.Teacher(architecture=[4,256,256,8],scale=[1,1,0,0])
```
## Methods
### generate_input
```python
generate_input(batch_size=1)
```
returns a randomly generated input
#### Arguments
* **batch_size:** integer, returns numpy array of shape ```[batch size,input dimension] ```. Here ```input dimension``` is the first element of the global variable ```architecture```

### predict
```python
predict(x_input)
```
returns the predicted value for input ```x_input```
#### Arguments
* **x_input:** input, numpy array of shape ```[batch size,input dimension]```. The entire input array is evaluated as one batch. Here ```input dimension``` is the first element of the global variable ```architecture```

```python
save_weights(folder,W=None,b=None)
```
saves ```W``` and ```b``` (weights and biases) as csv files.
#### Arguments
* **folder:** directory to save weights and biases
* **W:** Weights to be saved (list of numpy arrays). If ```None```, defaults to global variable ```W```
* **b:** Biases to be saved (list of numpy arrays). If ```None```, defaults to global variable ```b```
