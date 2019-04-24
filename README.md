# Classification de voitures

Des adolescents a créé un modèle qui reconnaît des voitures parmi plusieurs marques : BMW, Ferrari, Lamborghini et McLaren.
Ils ont utilisé les modules Tensorflow et Keras en python pour créer leur réseau de neurones.

## Stage IA à Magic Makers

[Magic Makers](https://www.magicmakers.fr/) propose des ateliers de programmation créative pour des jeunes de 7 à 15 ans. Depuis 2018, des ateliers pour adolescents autour de l'intelligence artifielle sont donnés durant les vacances. Lors du stage, les makers découvrent ce qu'est un réseaux de neurones et les notions s'y attachant (perceptron multi-couches, convolutions, overfit, etc) en créant des projets comme celui-ci !

## Auteur du projet

Ce projet a été réalisé par **Jorys et Maxence** (en 2nde) lors du stage de Juillet dans le centre de Magic Makers Vincennes, animé par **Romain et Jade**.


### Dataset

* Photos de voitures sélectionnées sur Google Image


### Entraînement

Jorys et Maxence ont commencé par sélectionner les photos de ses voitures pour leur modèle. Ils ont ensuite utilisé un réseau de neurones par convolution pour leur projet.

```
python3 train-cars.py
```
## Built With

* [Keras](https://keras.io/) - pour créer le modèle (avec TensorFlow)
* [Flask](http://flask.pocoo.org/) - pour créer une webapp
* [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html) - pour manipuler des images
* [Numpy](https://www.numpy.org/) - pour manipuler des tableaux
* [H5py](https://www.h5py.org/) - pour sauvegarder le modèle
* [Sklearn](https://scikit-learn.org/stable/) - pour mélanger et séparer les données

## Résultats

< à venir >

### Application

Une fois leur modèle entraîné, Jorys et Maxence ont créé un programme pour prédire la voiture présente sur une photo !

```
python3 predictor-cars.py
```

### Remerciement

* Merci à [Magic Makers](https://www.magicmakers.fr/)
* Merci à [Keras](https://keras.io/) pour faciliter la création de réseaux de neurones !
