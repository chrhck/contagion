# Contagion

Authors:

1. Christian Haack
2. Martina Karl
3. Stephan Meighen-Berger
4. Dominik Scholz
5. Andrea Turcati

## Table of contents

[[_TOC_]]

## Introduction

A python package to simulate the spread of diseases in a population.
The code is structured modular, allowing for easy extension/change
of the steps required in modelling the spread. The model is similar to the
analytic SIR model and it's extensions. Switching between these models
is done by setting the corresponding parameters for the infection.

## Model

The scial model is structed as
![Sketch of the model](images/Model_Basic.png)

The epidemiological model is based on a SEIR model.

## Code Example

A basic running example to interface with the package

```python
# Importing the package
from contagion import Contagion, config
# Creating contagion object
contagion = Contagion()
# Optional: Some settings
config['population size'] = 1000
# Runing the simulation
contagion.sim()
# Access simulation results using
results = contagion.statistics
```

## Code structure

On overview of the code structure
![Sketch of the model](images/Code_Structure.png)
