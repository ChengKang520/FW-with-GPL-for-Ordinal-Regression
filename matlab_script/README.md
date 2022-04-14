
Move and replace the M files: DAGNetwork.m and Trainer.m to file path mynnet\cnn\+nnet\+internal\+cnn if you are using Matlab2019b.
In Trainer.m from line 158 to line 294, the function:
function [net FuzzyValue] = trainFuzzy(this, net, data, windows)

and in DAGNetwork.m from line 870 to line 1123, the function:
```c
function [gradients, predictions, states, inputGradients] = computeGradientsForTrainingFuzzy( ...
        this, X, Y, propagateState, windows, dLossdOutput)

```

