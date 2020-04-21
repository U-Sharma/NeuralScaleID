# NeuralScaleID
## Model class (NeuralNet_Utkarsh.py)
you can instantiate a model via:
```python
import NeuralNet_Utkarsh as NN
model = NN.Model(architecture=[10,64,64,64,2],loss='KL',softmax=True)
```
### Methods
#### train
```python
train(x,y,batch_size=32,epochs=1,learning_rate=0.001,lr_scheduler = None,verbose=2,asymp_steps=1000,skip=100)
```
trains the model
