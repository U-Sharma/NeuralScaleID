# NeuralScaleID
This codebase implements teacher/student training with a variety of network sizes and shapes.  It also tests performance of CNNs on image datasets. The reader is strongly encouraged to refer to the example.ipynb in the codebase. More details of the codebase can be found in Documentation.md.

For the sake of organization, the python files for individual experients have been placed together in the folder Experiments. To run an experiment file that imports NeuralNet or teacher, it has to be in the same directory as those two modules. The organization of files in folder Experiments with respect to the paper is:
* KL.py: Cross-Entropy loss
* L2.py: Mean Squared Error loss
* GeneralizedLoss.py: Generalized loss 
* XxY.py: Product data manifold

The names of image dataset files are self explanatory

### Description of Teacher/Student modules
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

We can evaluate the loss on the trained model
```python
x_test = t.generate_input(batch_size=100)
y_test = t.predict(x_test)

test_loss = model.evaluate(x_test,y_test)
```

The teacher/student experiments have been done by first generating a database from teacher, and then loading the database in order to train the student (refer example.ipynb). In order to run those experiments, a database needs to be generated. Refer to the individual experiment python files for details of database.
