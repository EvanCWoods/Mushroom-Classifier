# Mushroom-Classifier

This repository includes all code used in the creation of a neural network application that classifies mushrooms as poisonous or not based on a set of visual factors. 

The files in the main branch demonstrate the building of the final network which has its hyper-parameters fine tuned to the problem and achieves an accuracy of greater than 98% on unseen testing data.

The project started with data exploration which can be found in the 'data-exploration.py' file. This gave insights into the data type, shape and usefulness of each data attribute.

The first model provides a benchmark for results to grow from. In the 'benchmark-model.py' file, the base model is found, consisting of only one hidden layer and 30 neurons. From here, tuning hyper-parameters is possible.

The heavy lifting in model architecture is done by the keras-tuner library in the 'NN-keras-tuner.py' file, which allows a developer to run through a sequence of hyper-parameter combinations and find the best one.

The final model found in the 'final-model.py' file includes all of the work done to this point, the correct preprocessing, data inclusions and transformations, model architecture and hyper-parameters are included in this model, and the accuracy of the model is plotted after running.


## Future Developments:
This project can be extended in effectiveness by implamenting it in a website that allows a user to upload a picture of a mushroom in the wild and get a classification for whether or not it is safe to eat.
