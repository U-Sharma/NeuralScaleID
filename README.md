# NeuralScaleID
This codebase implements teacher/student training with a variety of network sizes and shapes.  It also tests performance of CNNs on image datasets. The reader is strongly encouraged to refer to the example.ipynb in the codebase.

Generating a random teacher with 4 inputs and 2 outputs (softmax), and 2 hidden layers of width 128:
```python
import teacher
t = teacher.Teacher(architecture=[4,128,128,2],softmax=True)
```
Argument ```scale``` can be used to control the number of features. The inputs and scale are multiplied element wise before passing through the network. In the example below, the first two input inputs are multiplied by 1 and the last two multiplied by 0. Thus, this example has 2 input features.

```python
import teacher
t = teacher.Teacher(architecture=[4,128,128,2],softmax=True,scale=[1,1,0,0])
```

The module NeuralNet can be used to generate a fully connected neural network.
```python
import NeuralNet as student
model = student.Model(architecture=[4,32,32,32,2])
```
This model can be trained on the teacher 
```python
x_train = t.generate_input(batch_size=32*10000)
y_train = t.predict(x_train)

history = model.train(x_train,y_train)
```
