# NeuralScaleID
This codebase implements teacher/student training with a variety of network sizes and shapes.  It also tests performance of CNNs on image datasets. The reader is strongly encouraged to refer to the example.ipynb in the codebase.

Generating a random teacher with 4 inputs and 2 outputs (softmax), and 2 hidden layers of width 128:
```\python
import teacher
t = teacher.Teacher(architecture=[4,128,128,2],softmax=True)
```
Argument ```scale``` can be used to control the number of features. The inputs are multiplied element wise by the scale before passing through the network. In the example below, the first two input inputs are multiplied by 1 and the last two multiplied by 0. Thus, this example has 2 input features.

```\python
import teacher
t = teacher.Teacher(architecture=[4,128,128,2],softmax=True,scale=[1,1,0,0])
```
