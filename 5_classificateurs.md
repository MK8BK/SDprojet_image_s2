---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "afbf34e64e7471cefc07d9483063e7a8", "grade": false, "grade_id": "cell-3876f910a24fe8a7", "locked": true, "schema_version": 3, "solution": false}}

# Classificateurs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f1092a8183464dd2e0b498c1337d6626", "grade": false, "grade_id": "cell-65c63f4be1820d2e", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans cette feuille, nous allons explorer l'utilisation de plusieurs
classificateurs sur l'exemple des pommes et des bananes. Vous pourrez
ensuite les essayer sur votre jeu de données.

Commencons par charger les utilitaires et autres librairies:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b2e17e47ccdae1d5732b95b0d7416e66
  grade: false
  grade_id: cell-a2ccd1e13b05762c
  locked: true
  schema_version: 3
  solution: false
  task: false
---
import os, re
from glob import glob as ls
import numpy as np                    # Matrix algebra library
import pandas as pd                   # Data table (DataFrame) library
import seaborn as sns; sns.set()      # Graphs and visualization library
from PIL import Image                 # Image processing library
import matplotlib.pyplot as plt       # Library to make graphs 
# Configuration intégration dans Jupyter
%matplotlib inline

## les utilitaires
%load_ext autoreload
%autoreload 2
from utilities import *

## les jeux de données
from intro_science_donnees import data
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "d880dfccac7e7bb362d70f774bf09ebb", "grade": false, "grade_id": "cell-875bebae978f4f5a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Chargement et préparation des données

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b65b91941f8d669592af300673a8c5e9", "grade": false, "grade_id": "cell-ce0810dd4273ea24", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On charge le jeu de données prétraité (attributs rougeur et élongation
et classes des fruits), tel que fournis en semaine 3:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 769fa816745ea2782f19614eb432c19e
  grade: false
  grade_id: cell-e6d48e8b35c2d42a
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df = pd.read_csv("../Semaine3/attributs.csv", index_col=0)
# standardisation
dfstd =  (df - df.mean()) / df.std()
dfstd['class'] = df['class']
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "5e412381817b3618b8a7a7ca91094d61", "grade": false, "grade_id": "cell-24210bafac4b3490", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On partitionne le jeu de données en ensemble de test et
d'entraînement:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 5d2b328c09ea561ef6c95a69314be2ba
  grade: false
  grade_id: cell-00b3f36097ec9704
  locked: true
  schema_version: 3
  solution: false
  task: false
---
X = dfstd[['redness', 'elongation']]
Y = dfstd['class']
#partition des images
train_index, test_index = split_data(X, Y, seed=0)

#partition de la table des attributs
Xtrain = X.iloc[train_index]
Xtest = X.iloc[test_index]
#partition de la table des étiquettes
Ytrain = Y.iloc[train_index]
Ytest = Y.iloc[test_index]
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "741881738d7829a3b3d62de9cceb02f2", "grade": false, "grade_id": "cell-9e3cf5dde27fbb00", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Classificateurs basés sur les exemples (*examples-based*)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "45db778c1166c64282fd7d8e20a03d1b", "grade": false, "grade_id": "cell-f7f11edbe1e13a7d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous allons maintenant voir comment appliquer des classificateurs
fournis par la librairie `scikit-learn`.  Commençons par le
classificateur plus proche voisin déjà vu en semaines 3 et 4.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "40979d861a5204582bc97eca2e359c92", "grade": false, "grade_id": "cell-28f37cb9aba3b0d6", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### KNN : $k$-plus proche voisins

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: c23d3cd77a415c7fceb36d690329ec61
  grade: false
  grade_id: cell-786fada559c8c5e0
  locked: true
  schema_version: 3
  solution: false
  task: false
---
from sklearn.neighbors import KNeighborsClassifier

#définition du classificateur, ici on l'appelle classifier
# on choisit k=1
classifier = KNeighborsClassifier(n_neighbors=1)
# on l'ajuste aux données d'entrsainement
classifier.fit(Xtrain, Ytrain) 
# on calcule ensuite le taux d'erreur lors de l'entrainement et pour le test
Ytrain_predicted = classifier.predict(Xtrain)
Ytest_predicted = classifier.predict(Xtest)
# la fonction error_rate devrait etre présente dans votre utilities.py (TP3), sinon ajoutez-la
e_tr = error_rate(Ytrain, Ytrain_predicted)
e_te = error_rate(Ytest, Ytest_predicted)

print("Classificateur: 1 Neighrest Neighbor")
print("Training error:", e_tr)
print("Test error:", e_te)
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7c27b6028c9284dfe4422c81b3f7c658", "grade": true, "grade_id": "cell-4db77bd51d290099", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

**Exercice :** Quels sont les taux d'erreur pour l'ensemble
d'entraînement et l'ensemble de test ?

Les taux d'erreurs pour l'ensemble d'entrainement et de test sont repsectivement O% et 20%

On mémorise ces taux dans une table `error_rates` que l'on complétera
au fur et à mesure de cette feuille:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: d5bb250cb3310ca7ac9e6af12c58ca2d
  grade: false
  grade_id: cell-84331f5cb22a0f87
  locked: true
  schema_version: 3
  solution: false
  task: false
---
error_rates = pd.DataFrame([], columns=['entrainement', 'test'])
error_rates.loc["1 Neighrest Neighbor",:] = [e_tr, e_te]
error_rates
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1dd2e40abcbbaec8ad5dbd190a66d33a", "grade": false, "grade_id": "cell-2034d3a7e9dbbfad", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Fenêtres de Parzen (*Parzen window* ou *radius neighbors*)

Pour ce classificateur, on ne fixe pas le nombre de voisins mais un
rayon $r$; la classe d'un élément $e$ est prédite par la classe
majoritaire parmi les éléments de l'ensemble d'entraînement dans la
sphère de centre $e$ et de rayon $r$.

**Exercice :** Complétez le code ci-dessous:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 5f1723f468796a617b6b6d013f4f5a0d
  grade: false
  grade_id: cell-64dfc640f023f84e
  locked: false
  schema_version: 3
  solution: true
  task: false
---
from sklearn.neighbors import RadiusNeighborsClassifier
classifier = RadiusNeighborsClassifier(radius=1.0)

# on l'ajuste aux données d'entrainement

classifier.fit(Xtrain,Ytrain)

# on calcule ensuite le taux d'erreur lors de l'entrainement et pour le test

Ytrain_predicted = classifier.predict(Xtrain)
Ytest_predicted = classifier.predict(Xtest)
# on calcule les taux d'erreurs

e_tr2 = error_rate(Ytrain, Ytrain_predicted)
e_te2 = error_rate(Ytest, Ytest_predicted)

print("Classificateur: Parzen Window")
print("Training error:", e_tr2)
print("Test error:", e_te2)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1e18644108e320442bc0b8634f81b21e", "grade": false, "grade_id": "cell-540f1e8f9850f69f", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Complétez la table `error_rates` avec ce modèle, en
rajoutant une ligne d'index `Parzen Window`.

**Indication :** Utiliser `.loc` comme ci-dessus.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 6f6dbcaff1626d6fa8527a7ad4a00d14
  grade: false
  grade_id: cell-4274f47e142d9ed1
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
error_rates.loc["Parzen Window",:] = [e_tr2, e_te2]
error_rates
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: da22505d351571e98b45b6f44159e00d
  grade: true
  grade_id: cell-6ac30547e91d336f
  locked: true
  points: 2
  schema_version: 3
  solution: false
  task: false
---
assert isinstance(error_rates, pd.DataFrame)
assert list(error_rates.columns) == ['entrainement', 'test']
assert list(error_rates.index) == ['1 Neighrest Neighbor', 'Parzen Window']
assert (0 <= error_rates).all(axis=None), "Les taux d'erreurs doivent être positifs"
assert (error_rates <= 1).all(axis=None), "Les taux d'erreurs doivent être inférieur à 1"
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "bd250040d987603d47cd207402729cc1", "grade": true, "grade_id": "cell-7ca6c7b760e2f6d3", "locked": false, "points": 3, "schema_version": 3, "solution": true, "task": false}}

**Exercice $\clubsuit$ :** Faites varier le rayon $r$. Comment le taux
d'erreur varie-t-il ? Vous pouvez ajouter des modèles à la table
`error_rates` s'ils vous semblent pertinents

On remarque que le taux d'erreur à l'entrainement et au test augmente lorsqu'on utilise le rayon dans le modèle de Parzen.

```{code-cell} ipython3
for i in range(2,11):
    classifier = RadiusNeighborsClassifier(radius=i)
    # on l'ajuste aux données d'entrainement
    classifier.fit(Xtrain,Ytrain)
    # on calcule ensuite le taux d'erreur lors de l'entrainement et pour le test
    Ytrain_predicted = classifier.predict(Xtrain)
    Ytest_predicted = classifier.predict(Xtest)
    # on calcule les taux d'erreurs
    e_tr2 = error_rate(Ytrain, Ytrain_predicted)
    e_te2 = error_rate(Ytest, Ytest_predicted)
    error_rates.loc[f"Parzen Window radius={i}",:] = [e_tr2, e_te2]
```

```{code-cell} ipython3
error_rates
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "11aa8b1bee7bca0e0941f5be75907e9b", "grade": false, "grade_id": "cell-7474d5437867f93b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Classificateurs basés sur les attributs (*feature based*)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7f142dd64f1cac9b6487260ca1cff01d", "grade": false, "grade_id": "cell-bc351a36f5734e8b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Régression linéaire

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b6b80b5d2c6aa80c8a5ad69b4bddf7fb", "grade": true, "grade_id": "cell-aa0635012d566877", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

**Exercice :** Pourquoi ne peut-on pas appliquer la méthode de
régression linéaire pour classer nos pommes et nos bananes ?

Parce que nos pommes et nos bananes ne sont pas séparées par une droite.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6e75b4ffc13642cda57f2610ee0b93dc", "grade": false, "grade_id": "cell-f3fe417cf16e9bb2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Arbres de décision

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ca97a3e6e5fb381ffa0cf367ac37dd72", "grade": false, "grade_id": "cell-0c0e203b37439b49", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Les arbres de décison correspondent à des modèles avec des décisions
imbriquées où chaque noeud teste une condition sur une variable.  Les
étiquettes se trouvent aux feuilles.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "65228b363cf73812186190baeb789507", "grade": false, "grade_id": "cell-da133a018c6e8417", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Complétez le code ci-dessous.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: b706dd669f71f720a937332056ae3b4e
  grade: false
  grade_id: cell-8e19513e3f66e21a
  locked: false
  schema_version: 3
  solution: true
  task: false
---
from sklearn import tree

decision_tree = tree.DecisionTreeClassifier()
# on l'ajuste aux données d'entrainement

decision_tree.fit(Xtrain,Ytrain)
# on calcule ensuite le taux d'erreur lors de l'entrainement et pour le test

Ytrain_predicted = decision_tree.predict(Xtrain)
Ytest_predicted = decision_tree.predict(Xtest)

# on calcule les taux d'erreurs

e_tr = error_rate(Ytrain, Ytrain_predicted)
e_te = error_rate(Ytest, Ytest_predicted)


print("Classificateur: Arbre de decision")
print("Training error:", e_tr)
print("Test error:", e_te)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "7349afd1535a78ef980d8f6760875d5d", "grade": false, "grade_id": "cell-d818ab75031ccf3c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Complétez la table `error_rates` avec ce modèle.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: c939e2d995ec9bb6ca39f2e93265c069
  grade: false
  grade_id: cell-b2218cbf456bd91c
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
error_rates.loc["Tree decision",:] = [e_tr, e_te]
error_rates
print(error_rates)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 263cc1fd48d9747bbe78937fdf9a3472
  grade: true
  grade_id: cell-57894376c13e530a
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert isinstance(error_rates, pd.DataFrame)
assert list(error_rates.columns) == ['entrainement', 'test']
assert error_rates.shape[0] >= 3
assert (0 <= error_rates).all(axis=None), "Les taux d'erreurs doivent être positifs"
assert (error_rates <= 1).all(axis=None), "Les taux d'erreurs doivent être inférieur à 1"
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4fa7e91b5a9e8184312ecfe62340a032", "grade": false, "grade_id": "cell-f1f775a2ccb77c41", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Représentez l'arbre de décision comme vu lors du CM5.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 7b63b16ea19a3384fae1cf81c25b7b25
  grade: false
  grade_id: cell-e4ce3f1406e9fe61
  locked: false
  schema_version: 3
  solution: true
  task: false
---
import matplotlib.pyplot as plt

plt.figure(figsize=(12,12)) 
tree.plot_tree(decision_tree, fontsize=10) 
plt.show()
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4feff09eb4df25c0948e4e4a799028b6", "grade": true, "grade_id": "cell-316a87fad1287712", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false}}

**Exercice :** Interprétez cette figure.

Cette figure représente l'arbre de décision de notre modèle.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4d130da1537149b2d938f87bb679c7c1", "grade": false, "grade_id": "cell-ff5d2dccfd4a6505", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Perceptron

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e493f04249e130eb23a4e0d6b7f05a98", "grade": false, "grade_id": "cell-e4a08d06646c875e", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Le perceptron est un réseau de neurones artificiels à une seule couche
et donc avec une capacité de modélisation limitée; pour le problème
qui nous intéresse cela est suffisant. Pour plus de détails, revenez
au [cours](CM5.md)

**Exercice :** Complétez le code ci-dessous, où l'on définit un modèle
de type `Perceptron` avec comme paramètres $10^{-3}$ pour la tolérence, $36$ pour l'état aléatoire (*random state*) et 100 époques (*max_iter*)

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 5c9287f84d7c30ac9d4d477e2d8959db
  grade: false
  grade_id: cell-c6bfeee5a4a966cc
  locked: false
  schema_version: 3
  solution: true
  task: false
---
from sklearn.linear_model import Perceptron

# définition du modèle de classificateur
perceptron = Perceptron(tol=1e-3, random_state=36, max_iter=100)
# on l'ajuste aux données d'entrainement

perceptron.fit(Xtrain,Ytrain)
# on calcule ensuite le taux d'erreur lors de l'entrainement et pour le test

Ytrain_predicted = perceptron.predict(Xtrain)
Ytest_predicted = perceptron.predict(Xtest)

# on calcule les taux d'erreurs

e_tr = error_rate(Ytrain, Ytrain_predicted)
e_te = error_rate(Ytest, Ytest_predicted)

print("Classificateur: Perceptron")
print("Training error:", e_tr)
print("Test error:", e_te)
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9976c26d3d2c1e44c76a11b3d7d6b95a", "grade": true, "grade_id": "cell-f8994015ecc492d3", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

**Exercice :** Lisez la documentation de `Perceptron`. À quoi
correspond le paramètre `random_state` ?

Le paramètre random_state correspond au mélange de l'ensemble d'entrainement lorsque l'attribut 'shuffle' est True.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 0600469ba02818badb43e47910bf271e
  grade: false
  grade_id: cell-4f0cec434737cda7
  locked: false
  schema_version: 3
  solution: true
  task: false
---
help("sklearn.linear_model.Perceptron")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4913e916d7d45477247b2487260702b3", "grade": false, "grade_id": "cell-65470533b4335e5a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Complétez la table `error_rates` avec ce modèle.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: b5a8399a63c996a0984792b535fa4065
  grade: false
  grade_id: cell-cfe370c6b70f6d27
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
error_rates.loc["Perceptron",:] = [e_tr, e_te]
error_rates
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 49ed7a76afdf1c5075ac8514f69c24b5
  grade: true
  grade_id: cell-b72fa191a2a82889
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert error_rates.shape[0] >= 4
assert error_rates.shape[1] == 2
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "760dec58453dfd0d440baf2c8de3bdc9", "grade": false, "grade_id": "cell-b546e3375477b608", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## $\clubsuit$ Points bonus : construction du classificateur «une règle» (*One Rule*)

Faites cette partie ou bien passez directement à la conclusion.

En complétant le code ci-dessous, créez votre premier classificateur
qui:
* sélectionne le "bon" attribut (rougeur ou élongation pour le
  problème des pommes/bananes), appelé $G$ (pour *good*). C'est
  l'attribut qui est le plus corrélé (en valeur absolue, toujours !)
  aux valeurs cibles $y = ± 1$;
* détermine une valeur seuil (*threshold*);
* utilise l'attribut `G` et le seuil pour prédire la classe des éléments.
        
Un canevas de la classe `OneRule` est fournit dans la classe
`utilities.py`; vous pouvez le compléter ou bien la programmer
entièrement vous même.

Ce classificateur est-il basé sur les attributs ou sur les exemples?

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 9f7385d47625cbe5517b632691a023d4
  grade: false
  grade_id: cell-b72b2907842cb7fb
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Use this code to test your classifier
classifier = OneRule()
classifier.fit(Xtrain, Ytrain) 
Ytrain_predicted = classifier.predict(Xtrain)
Ytest_predicted = classifier.predict(Xtest)
e_tr = error_rate(Ytrain, Ytrain_predicted)
e_te = error_rate(Ytest, Ytest_predicted)
print("Classificateur: One rule")
print("Training error:", e_tr)
print("Test error:", e_te)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "63825ae948ab016d3e3be1b480b0c325", "grade": false, "grade_id": "cell-472a8c23ef803ec9", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Complétez la table `error_rates` avec ce modèle.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 9f9fbca044d9d5ca0de79a289e657a64
  grade: false
  grade_id: cell-7b189e436af168e3
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
print(error_rates)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: d9cc4e307aa25ea9db7429b85c63e193
  grade: true
  grade_id: cell-14c847f98a88a373
  locked: true
  points: 0
  schema_version: 3
  solution: false
  task: false
---
assert error_rates.shape[0] >= 5
assert error_rates.shape[1] == 2
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 9eac0c359d449ad1c0026e38854320af
  grade: false
  grade_id: cell-587a9581023e668d
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# On charge les images
dataset_dir = os.path.join(data.dir, 'ApplesAndBananasSimple')
images = load_images(dataset_dir, "*.png")
# This is what you get as decision boundary.
# The training examples are shown as white circles and the test examples are blue squares.
make_scatter_plot(X, images.apply(transparent_background_filter),
                  [], test_index, 
                  predicted_labels='GroundTruth',
                  feat = classifier.attribute, theta=classifier.theta, axis='square')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e9f6c1a409234b6d29ce5f38de89a7bf", "grade": false, "grade_id": "cell-1b8779ac28a26c59", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Comparez avec ce que vous auriez obtenu en utilisant les 2 attributs avec le même poids lors de la décision de classe.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 5f6db3de94731a2fbf4749d55449b82f
  grade: false
  grade_id: cell-d0e8bd2a9132fea6
  locked: true
  schema_version: 3
  solution: false
  task: false
---
make_scatter_plot(X, images.apply(transparent_background_filter),
                  [], test_index, 
                  predicted_labels='GroundTruth',
                  show_diag=True, axis='square')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8db48dc81f852cf17c594ed3396acf42", "grade": false, "grade_id": "cell-28e6ef8e4fa12a3c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Conclusion

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "251fa35c1a5f4e7d3c1c7d27b1c50648", "grade": true, "grade_id": "cell-1b50f7767c53d39b", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false}}

**Exercice :** Comparez les taux d'erreur et d'entraînement de vos
différents classificateurs pour le problème des pommes et des bananes.

VOTRE RÉPONSE ICI

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "623cd73082a3b65a692ad8b02da7e0ba", "grade": false, "grade_id": "cell-466bf7addb6e465a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans cette feuille vous avez découvert comment utiliser un certain
nombre de classificateurs, voire comment implanter le vôtre, et
comment jouer sur les paramètres de ces classificateurs (par exemple
la tolérance du perceptron ou le nombre de voisins du KNN) pour
essayer d'optimiser leur performance.

Mettez à jour votre rapport et déposez votre travail.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3e3d916a7c8ce54f4dc14284f4de3fc4", "grade": false, "grade_id": "cell-466bf7addb6e465b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vous êtes maintenant prêts pour revenir à votre [analyse de
données](4_analyse_de_donnees.md) pour mettre en œuvre ces
classificateurs sur votre jeu de données.

**Dans le projet 1, on vous demande de choisir un seul classificateur,
ainsi que ses paramètres**.  Nous verrons dans la seconde partie de
l'UE comment comparer systématiquement les classificateurs.
