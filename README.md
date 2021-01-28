# adversarial-generator

A simple app generating [adversarial examples](https://openai.com/blog/adversarial-example-research/) with a few methods, for various neural network based image classifiers.

![screenshot](https://i.imgur.com/ZDGIr4D.png)

## Project structure

The project consists of two main modules:

- models - this one contains all ML-related stuff
- app - Django application that provides UI for the generator

## Run pylint

`find . -type f -name "*.py" | xargs pylint `
