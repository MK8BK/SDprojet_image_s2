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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e877a7f5aa56b85ddc209ecbb2d54445", "grade": false, "grade_id": "cell-3876f910a24fe8a7", "locked": true, "schema_version": 3, "solution": false}}

# VI-ME-RÉ-BAR sur vos propres données!

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "22fb72602440facdad988182f9c6695d", "grade": true, "grade_id": "cell-3b25fe1716dd8293", "locked": false, "points": 3, "schema_version": 3, "solution": true, "task": false}}

**Instructions**:
- Vous effacerez les instructions au fur et à mesure que vous les
  aurez suivies. Commencez par effacer celle-ci!
- Mettez ici une description de votre jeu de données: lequel avez vous
  choisi, quel est le défi? Intuitivement quels critères pourraient
  permettre de distinguer les deux classes d'images?

+++

0 et 1

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 6b6e1663d39b90bc4820816527aa31d0
  grade: false
  grade_id: cell-638b8591a86d2f35
  locked: true
  schema_version: 3
  solution: false
  task: false
---
%load_ext autoreload
%autoreload 2
from utilities import *
from intro_science_donnees import data
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 30662a4cdfa31ad7e09d600c6fd89de6
  grade: false
  grade_id: cell-f463237384c14d8c
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Load general libraries
import os, re
from glob import glob as ls
import numpy as np                    # Matrix algebra library
import pandas as pd                   # Data table (DataFrame) library
import seaborn as sns; sns.set()      # Graphs and visualization library
from PIL import Image                 # Image processing library
import matplotlib.pyplot as plt       # Library to make graphs 
# Command to insert the graphs in line in the notebook:
%matplotlib inline

# Reload code when changes are made
%load_ext autoreload
%autoreload 2

# Import utilities
from utilities import *
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b5bcf7ce90464828259054e0f8d258ce", "grade": false, "grade_id": "cell-1e377d9f288ab8e0", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Étape 1: prétraitement et [VI]sualisation

+++

Le jeu de données consiste en les images suivantes:

**Instruction :** Chargez votre jeu de données comme dans la feuille
`3_jeux_de_donnees.md` de la semaine dernière, en stockant les
images dans la variables `images` et en les affichant.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: cd37d1d6f3c86eb99de8def1f4bb7ce3
  grade: false
  grade_id: cell-596455595966feb5
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
dataset_dir = os.path.join(data.dir, 'ZeroOne')
images = load_images(dataset_dir, "*.png")
image_grid(images, titles=images.index)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: a2d30e6e80157634ba3fbd15f8eab118
  grade: true
  grade_id: cell-d3b1c03f1fbc66c4
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert isinstance(images, pd.Series)
assert len(images) == 20
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b750e129defcac1b2774f8d55bd0bfa3", "grade": false, "grade_id": "cell-1ad9ea64fc9cfd94", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Prétraitement

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b3cc5b3152415f80c66057a4ac36796d", "grade": false, "grade_id": "cell-c5eeea84f126a6a9", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Les données sont très souvent prétraitées c'est-à-dire **résumées
selon différentes caractéristiques** : chaque élément du jeu de
données est décrit par un ensemble [**d'attributs**](https://en.wikipedia.org/wiki/Feature_(machine_learning))
-- propriétés ou caractéristiques mesurables de cet élément ; pour un
animal, cela peut être sa taille, sa température corporelle, etc.

C'est également le cas dans notre jeu de données : une image est
décrite par le couleur de chacun de ses pixels. Cependant les pixels
sont trop nombreux pour nos besoins. Nous voulons comme la semaine
dernière les remplacer par quelques attributs mesurant quelques
propriétés essentielles de l'image, comme sa couleur ou sa forme
moyenne: ce sont les données prétraitées.

La semaine dernière, les données prétraitées vous ont été fournies
pour les pommes et les bananes.
Cette semaine, grâce aux trois feuilles précédentes, vous avez les
outils et connaissances nécessaires pour effectuer le prétraitement 
directement vous-même:

- la feuille de rappel sur la [gestion de tableaux](1_tableaux.md); 
- la feuille sur le [traitement des images](2_images.md);
- la feuille sur l'[extraction d'attributs](3_extraction_d_attributs.md).

Pour commencer, la table prétraitée contient les attributs `redness`
et `elongation` -- tels que vous les avez défini dans la feuille
[extraction d'attributs](3_extraction_d_attributs.md) -- appliqués à
votre jeu de données":

+++

 utiliser `foreground_filter()` dans le cas ou les images sont bruyantes ou non monochromes

```{code-cell} ipython3
---
code_folding: []
deletable: false
nbgrader:
  cell_type: code
  checksum: 5a09476d767f2b3b5cbf39f920ceb6ee
  grade: false
  grade_id: cell-4b826c34cfe02997
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
sample_images = load_images("blablabla", "*.png")
thresh_images = [foreground_filter(img) for img in sample_images]
image_grid(thresh_images, titles=images.index)
```

```{code-cell} ipython3
from PIL import Image 
import PIL 
  
# creating a image object (main image) 
for i in range(len(images)): 
    images[i].save(images.index[i])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "cf0e189af62197b241030a88f713c749", "grade": false, "grade_id": "cell-3ea685b93ca235c7", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": true}}

**Exercice :**
1. Implémentez dans `utilities.py` de nouveaux attributs adaptés à votre jeu de données. Si vous en avez besoin, vous pouvez utiliser les cellules ci-dessous voire en créer de nouvelles; sinon simplement videz les.

  **Indications**: vous pouvez par exemple vous inspirer
  - des attributes existants comme `redness`;
  - des exemples donnés dans le cours: *matched filter*, analyse en composantes principales (PCA).

+++

`elongation()` est deja un premier critere pour separer les 0 et les 1.

On implemente `boucle()` qui renvoie `true` si l'image en question (supposee noir sur blanc, contient une boucle.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 3691134a0bcfc2b8486069fec95621f8
  grade: false
  grade_id: cell-90320016ffc3a6b2
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
nimg = np.array(images[0])
nimg.shape
```

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 02bd6fd2b1c87e8f09796135a2f45237
  grade: false
  grade_id: cell-90320016ffc3a6b3
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
image_grid(images, 
           titles=["{0:.2f}".format(elongation(img)) for img in images])
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b7af4354b32a9e95a7a2c1182881127d", "grade": true, "grade_id": "cell-4a701722d7649d16", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

2. Comment les avez-vous choisis?

   VOTRE RÉPONSE ICI

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "aec697c54b9078ed166f363166e41e8f", "grade": false, "grade_id": "cell-d144e561c7a8d7d9", "locked": true, "schema_version": 3, "solution": false, "task": false}}

3. Ajoutez une colonne par attribut dans la table `df`, en conservant les précédents

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 1e6992a0256e6dd1f58f4b9b5da7bb43
  grade: false
  grade_id: cell-d933a673c19c7e6d
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "bb480abd9baac97518a69b7de64cf014", "grade": false, "grade_id": "cell-d635e7502cb91b2a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vérifications:
- la table d'origine est préservée:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 00cf3c65f455842623627fceda673e2d
  grade: true
  grade_id: cell-105a1d4f853b203c
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert len(df[df['class'] ==  1]) == 10
assert len(df[df['class'] == -1]) == 10
assert 'redness' in df.columns
assert 'elongation' in df.columns
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "450cedbf2975206641aa60b1b396769d", "grade": false, "grade_id": "cell-86b1db13ae3622b1", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- Nouveaux attributs:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b087dcf13b7252daecf1d1f8e2450318
  grade: true
  grade_id: cell-d8f0986d0b430798
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert len(df.columns) > 3, "Ajoutez au moins un attribut!"
assert df.notna().all(axis=None), "Valeurs manquantes!"
for attribute in df.columns[3:]:
    assert pd.api.types.is_numeric_dtype(df[attribute]), \
        f"L'attribut {attribute} n'est pas numérique"
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: acd89c5343b85c78b5bc14691087c6be
  grade: true
  grade_id: cell-03e40d6a322dfc9d
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert len(df.columns) > 4, "Gagnez un point en ajoutant un autre attribut"
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9ce3964950e281aa1cb29fdc7d21d090", "grade": false, "grade_id": "cell-592f7c21def71b06", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Standardisez les colonnes à l'exception de la colonne
`class`, afin de calculer les corrélations entre colonnes

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 75693d2c986d7eb3e5939bce97836887
  grade: false
  grade_id: cell-0c29581ba1da5a27
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
dfstd
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "11583edd1f5d181027fac76ecf2e729b", "grade": false, "grade_id": "cell-feea0a235f81712c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vérifions :

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 2e045804a7ee3835b8ebd6c668d16562
  grade: false
  grade_id: cell-69e2c8c203efb549
  locked: true
  schema_version: 3
  solution: false
  task: false
---
dfstd.describe()
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: c46020ebc9892ac98e0a366de26fbc7e
  grade: true
  grade_id: cell-b6120056f2331c33
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert dfstd.shape == df.shape
assert dfstd.index.equals(df.index)
assert dfstd.columns.equals(df.columns)
assert (abs(dfstd.mean()) < 0.01).all()
assert (abs(dfstd.std() - 1) < 0.1).all()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "720b9082c215b4223679b9f77a642dd3", "grade": false, "grade_id": "cell-16a95948fd5c0ef3", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Le prétraitement est terminé!

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "90a4b4e0ea8602608efc8be5f4b76809", "grade": false, "grade_id": "cell-043c3e7edafa0af9", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Visualisation

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3cba94907267a6b6afee5ba9fd083ff2", "grade": false, "grade_id": "cell-eb183dfd2fb60a40", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Extrayons quelques statistiques de base:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 32218974407dd7bddc4e5f6f6f47a298
  grade: false
  grade_id: cell-5adf965bf26113e8
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3cf82ca26d0187a3ba36dd851c4d6ed2", "grade": false, "grade_id": "cell-6c7b07fdc56a1970", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :**
- Visualisez le tableau de données sous forme de carte de chaleur (*heat map*):

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 04afb3a9abcdf59095b6662268bb26d3
  grade: false
  grade_id: cell-59b086b9a7351acb
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "271975688da40307bf7ecb528ab32e4c", "grade": false, "grade_id": "cell-dee09fc8d9185846", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- sa matrice de corrélation:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: f072ef0f33c30bdd71cd99d49d1e9242
  grade: false
  grade_id: cell-5d6f2f8188882c92
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2d168577f1b08233d34cc1ab6f07184b", "grade": false, "grade_id": "cell-6e71eda9745ba384", "locked": true, "schema_version": 3, "solution": false, "task": false}}

- ainsi que le nuage de points (*scatter plot*):

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: f50dbcef5c2cc4eb7db9eeb8c0d8766a
  grade: false
  grade_id: cell-10bd824917033e6a
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3a719c9648b336d7bc04e7b06661d4f8", "grade": false, "grade_id": "cell-bca3d5b71feb6ff1", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Observations

**Exercice :** Décrivez ici vos observations: corrélations apparentes
ou pas, interprétation de ces corrélations à partir du nuage de
points, etc. Est-ce que les attributs choisis semblent suffisants?
Quel attribut semble le plus discriminant? Est-ce qu'un seul d'entre
eux suffirait?

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6f529903b12f2180d2a775daba172e25", "grade": true, "grade_id": "cell-e22718dde2cfd4ba", "locked": false, "points": 4, "schema_version": 3, "solution": true, "task": false}}

VOTRE RÉPONSE ICI

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "780219ca7ddda552ea13fa1fd6609265", "grade": false, "grade_id": "cell-2668dce82b589f34", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Étape 2: [ME]sure de performance (*[ME]tric*)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b49f6d13a478e294be514a6488e347f3", "grade": false, "grade_id": "cell-2edc6abcf3a72dd2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Pour mesurer les performances de ce problème de classification, nous
utiliserons la même métrique par taux d'erreur que dans le TP3:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 618f2f555d926f5492c2441772f09723
  grade: false
  grade_id: cell-1fa6eb65d858c61c
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(error_rate)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "fca79a2271986cdbbefd31158166a6fb", "grade": false, "grade_id": "cell-fe6ef77774bf777c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Partition (*split*) du jeu de données en ensemble d'entraînement et ensemble de test

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1073ff388ab028c871e3d9ebde943c61", "grade": false, "grade_id": "cell-1a09d1e1733ada10", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Extraire, depuis `dfstd`, les deux attributs choisis dans `X` et la vérité terrain dans
`Y`:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 93b8c63a8e67ed7ce0d5999d175a97da
  grade: false
  grade_id: cell-fb5cd75f3de3a8a2
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

Ajouter un autotest que les attributs ne sont pas redness/elongation : un nouvel attribut ; deux nouveaux attributs

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: cac3960f3b06171eac77d75b29d25c2f
  grade: true
  grade_id: cell-666625af65184ae8
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert isinstance(X, pd.DataFrame), "X n'est pas une table Pandas"
assert X.shape == (20,2), "X n'est pas de la bonne taille"
assert set(X.columns) != {'redness', 'elongation'}
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 7abcef650ef1a7e6257ae0c6e20d3f0a
  grade: true
  grade_id: cell-6155db153be32033
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert 'redness' not in X.columns and 'elongation' not in X.columns, \
   "Pour un point de plus: ne réutiliser ni la rougeur, ni l'élongation"
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "853f80a58347bfa16eb8bdeb15c19d6c", "grade": false, "grade_id": "cell-672066b540b7e27b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Maintenant partitionnez l'index des images en ensemble
d'entraînement (`train_index`) et ensemble de test
(`test_index`). Récupérez les attributs et classes de vos images selon
l'ensemble d'entraînement `(Xtrain, Ytrain)` et celui de test `(Xtest,
Ytest)`.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 6dd6112bc416f09425e803631d28b8b2
  grade: false
  grade_id: cell-240f317e24d0e8c5
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 543dd26f1d5b2db6c2810087c5dabde0
  grade: true
  grade_id: cell-3be096fc97970fbc
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert train_index.shape == test_index.shape
assert list(sorted(np.concatenate([train_index, test_index]))) == list(range(20))

assert Xtest.shape == Xtrain.shape
assert pd.concat([Xtest, Xtrain]).sort_index().equals(X.sort_index())

assert Ytest.shape == Ytrain.shape
assert pd.concat([Ytest, Ytrain]).sort_index().equals(Y.sort_index())
assert Ytest.value_counts().sort_index().equals(Ytrain.value_counts().sort_index())
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "aaf3d8f5fb773034041687afa70348af", "grade": false, "grade_id": "cell-50c7b65a31fd47cb", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Affichez les images qui serviront à entraîner notre
modèle de prédiction (*predictive model*):

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: c7964ea91bb446bee9fe0fbc4e37b7ab
  grade: false
  grade_id: cell-ad5ca9b40a1f2cd1
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e64570807a43c88693069a74eaf99b57", "grade": false, "grade_id": "cell-031b7382abe186ef", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Affichez celles qui permettent de le tester et
d'évaluer sa performance:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 71b5600028c8a2fab07c08f146af65da
  grade: false
  grade_id: cell-1f3414a9f879a9d1
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a16f7f2ccdc41606de030e263f1c53f2", "grade": false, "grade_id": "cell-e44d2840e7481ec1", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Représentez les images sous forme de nuage de points en
fonction de leurs attributs:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 4682a59b4e3ed77249706a18b6ec264d
  grade: true
  grade_id: cell-c7dc96827c21b4b9
  locked: false
  points: 1
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "92fd26fbb3226b6054d55cbc4ca94dcb", "grade": false, "grade_id": "cell-ef6ffea3c9345a6f", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Taux d'erreur

Comme la semaine dernière, nous utiliserons le taux d'erreur comme
métrique, d'une part sur l'ensemble d'entraînement, d'autre part sur
l'ensemble de test. Implémentez la fonction `error_rate` dans votre
utilities.py. Pour vérifier que c'est correctement fait, nous
affichons son code ci-dessous:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 19048026419c75b7d2b389917743506b
  grade: false
  grade_id: cell-baa5e37a60edad22
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(error_rate)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0aad688914be7e65788713978ac3c2a8", "grade": false, "grade_id": "cell-2831ca6c9a3bfce5", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Étape 3: [RE]férence (*base line*)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "460a70846edc8dd32952b00dbe109551", "grade": false, "grade_id": "cell-9876b02ba8d2de55", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Classificateur

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0cad249e3b5505e8be4b7aae730a3735", "grade": true, "grade_id": "cell-5f7be836c3e06143", "locked": false, "points": 2, "schema_version": 3, "solution": true, "task": false}}

- En Semaine 4: faites la suite de cette feuille avec l'algorithme du
  plus proche voisin, comme en Semaine 3.

- En Semaine 5: faites la feuille sur les [classificateurs](../Semaine5/1_classificateurs.md)
  puis faites la suite de cette feuille avec votre propre classificateur,
  en notant au préalable votre choix de classificateur ici:

  VOTRE RÉPONSE ICI

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4fa688886d4cceaf1b6076e2720905e5", "grade": false, "grade_id": "cell-f541bcd7f0f3912d", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** 
Ci-dessous, définissez puis entraînez votre classificateur sur l'ensemble d'entraînement.

**Indication :** Si vous avez besoin de code supplémentaire pour cela, mettez-le dans `utilities.py`.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 553e412d52aa0920c309520ec1e30c96
  grade: false
  grade_id: cell-85205b5012588319
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9c3e81a24a4c69d62c580b683ae21664", "grade": false, "grade_id": "cell-991d8893ddaefe2b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Calculez les prédictions sur l'ensemble d'entraînement
et l'ensemble de test, ainsi que les taux d'erreur dans les deux cas:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 39e895304f018b8fc86b57dcaaf6bff1
  grade: false
  grade_id: cell-85205b5012588320
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
print("Training error:", e_tr)
print("Test error:", e_te)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: b11a6a34cb5c33b05b2e4a035785d6e4
  grade: true
  grade_id: cell-195ba103a7ccc55b
  locked: true
  points: 1
  schema_version: 3
  solution: false
  task: false
---
assert Ytrain_predicted.shape == Ytrain.shape
assert Ytest_predicted.shape == Ytest.shape
assert 0 <= e_tr and e_tr <= 1
assert 0 <= e_te and e_te <= 1
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "fae208e6bd19f0670dd8df95f6d64171", "grade": false, "grade_id": "cell-139252fc350603b2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Visualisons les prédictions obtenues:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: d52aa128366c4fd41fdcf88b6103e580
  grade: false
  grade_id: cell-fdaa23f990c0b172
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# The training examples are shown as white circles and the test examples are black squares.
# The predictions made are shown as letters in the black squares.
make_scatter_plot(X, images.apply(transparent_background_filter),
                  train_index, test_index, 
                  predicted_labels=Ytest_predicted, axis='square')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "00fb56afb030c89b52f77facddd94ba1", "grade": false, "grade_id": "cell-b0226451f00b5833", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Interprétation

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "dac1eac37628fb4e08a08a8bfd21afe5", "grade": true, "grade_id": "cell-b0226451f00b5834", "locked": false, "points": 5, "schema_version": 3, "solution": true, "task": false}}

**Exercice :** Donnez ici votre interprétation des résultats. La
performance des prédictions paraît elle satisfaisante? Avez vous une
première intuition de comment l'améliorer?

VOTRE RÉPONSE ICI

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8a2a785561a621cbac2dc3ca33e6a4ac", "grade": false, "grade_id": "cell-5539fbbf5565fb1b", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Étape 4: [BAR]res d'erreur (*error bar*)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "10804474205e34d74360e3d5a262c026", "grade": false, "grade_id": "cell-5539fbbf5565fb1c", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Barre d'erreur 1-sigma

**Exercice :** Comme première estimation de la barre d'erreur,
calculez la barre d'erreur 1-sigma pour le taux d'erreur `e_te`:

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 1146d04d0da826d62ebb6a72cec23351
  grade: false
  grade_id: cell-40a54824a74c8fdc
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# YOUR CODE HERE
raise NotImplementedError()
print("TEST SET ERROR RATE: {0:.2f}".format(e_te))
print("TEST SET STANDARD ERROR: {0:.2f}".format(sigma))
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "fe8f472d1b0d999fe0d8d39586805660", "grade": false, "grade_id": "cell-209765e9c10f8f32", "locked": true, "schema_version": 3, "solution": false, "task": false}}

### Barre d'erreur par validation croisée (Cross-Validation)

Nous calculons maintenant une autre estimation de la barre d'erreur en
répétant l'évaluation de performance pour de multiples partitions
entre ensemble d'entraînement et ensemble de test :

```{code-cell} ipython3
n_te = 10
SSS = StratifiedShuffleSplit(n_splits=n_te, test_size=0.5, random_state=5)
E = np.zeros([n_te, 1])
k = 0
for train_index, test_index in SSS.split(X, Y):
    print("TRAIN:", train_index, "TEST:", test_index)
    Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
    Ytrain, Ytest = Y.iloc[train_index], Y.iloc[test_index]
    neigh.fit(Xtrain, Ytrain.ravel()) 
    Ytrain_predicted = neigh.predict(Xtrain)
    Ytest_predicted = neigh.predict(Xtest)
    e_tr = error_rate(Ytrain, Ytrain_predicted)
    e_te = error_rate(Ytest, Ytest_predicted)
    print("TRAIN ERROR RATE:", e_tr)
    print("TEST ERROR RATE:", e_te)
    E[k] = e_te
    k = k+1
    
e_te_ave = np.mean(E)
# It is bad practice to show too many decimal digits:
print("\n\nCV ERROR RATE: {0:.2f}".format(e_te_ave))
print("CV STANDARD DEVIATION: {0:.2f}".format(np.std(E)))

sigma = np.sqrt(e_te_ave * (1-e_te_ave) / n_te)
print("TEST SET STANDARD ERROR (for comparison): {0:.2f}".format(sigma))
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "998da4c6471f630cec6a0395564210dc", "grade": false, "grade_id": "cell-df1f1b0b05e548d4", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Conclusion

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "11b535cf983332e0837c47fbc3a26668", "grade": false, "grade_id": "cell-df1f1b0b05e548d6", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Résumez ici les performances obtenues, tout d'abord
avec votre référence, puis avec les variantes que vous aurez explorées
en changeant d'attributs et de classificateur. Puis vous commenterez sur
la difficulté du problème ainsi que les pistes possibles pour obtenir
de meilleures performances, ou pour généraliser le problème.

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "989bef7b6f89f347f62e28548439cfad", "grade": true, "grade_id": "cell-8676a3839d9b5559", "locked": false, "points": 5, "schema_version": 3, "solution": true, "task": false}}

VOTRE RÉPONSE ICI

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "46043f1af5a2a64d6905b99242017ce0", "grade": false, "grade_id": "cell-df1f1b0b05e548d5", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice :** Complétez votre rapport (Semaine 4/Semaine5)
