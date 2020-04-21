# NeuralScaleID
## Model class (NeuralNet_Utkarsh.py)
you can instantiate a model via:
```python
import NeuralNet_Utkarsh as NN
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
* **layer:** Integer.1,2... for 1st,2nd... hidden layer If ```None```, returns a dictionary of all layer outputs. 
