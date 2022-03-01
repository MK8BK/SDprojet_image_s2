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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f6c8728c41191e222c533bab4bd2e89f", "grade": false, "grade_id": "cell-4911a792a82448d7", "locked": true, "schema_version": 3, "solution": false}}

# Semaine 5

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "dfb82db384cb593e909ba7ab81c241a5", "grade": false, "grade_id": "cell-4911a792a82448d8", "locked": true, "schema_version": 3, "solution": false}}

## Consignes

Avant mardi 1er mars à 22h:
- Relire les [diapos du cours 5](CM5.md)
- Rendu final du mini-projet 1. Vous devrez avoir déposé votre version
  finale du dossier `Semaine5` sur GitLab. Pour les binômes, chaque
  membre devra faire un dépôt séparé, même si les deux soumissions
  sont identiques.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e0c3a34bc4de1bb2f691a60706b3af12", "grade": false, "grade_id": "cell-4911a792a82448d9", "locked": true, "schema_version": 3, "solution": false}}

## TP: fin du mini-projet 1

La semaine précédente, vous avez appliqué le schéma VI-ME-RÉ-BAR --
[VI]sualisation, [MÉ]trique, [Ré]férence, [BAR]res d'erreur -- pour la
classification de votre propre jeu de données. Vous avez:

- Fait quelques essais avec un des jeux de données fournis
- Déroulé l'analyse de données avec des attributs simples (par exemple
  rougeur et élongation) ainsi qu'un premier classifieur, afin
  d'obtenir votre référence: sans chercher à optimiser, quelle
  performance de classification obtient-on?
- Implémenté d'autres attributs et déroulé à nouveau l'analyse pour
  obtenir une meilleure performance.

### Objectifs

L'objectif de cette deuxième semaine de mini-projet est de mettre en
application le [cours de cette semaine](CM5.md) en explorant d'autres
classificateurs. Vous pouvez au choix:

- Implémenter l'un des classificateurs décrits dans le cours:
  - OneR(**)
  - KNN: k-plus proches voisins (*)
  - Arbre de décision (*)
  - Fenêtre de Parzen (*)
  - Perceptron (**)
- Essayer plusieurs autres [classificateurs fournis par Scikit-Learn](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) comme par exemple:
  - Autre méthode de noyau (****)
  - Perceptron multi-couche (****)

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1f057cf3b9aeeb6586f22994d26f6597", "grade": false, "grade_id": "cell-4911a792a82448dD", "locked": true, "schema_version": 3, "solution": false}}

### Au travail!

- [ ] Vérifiez votre inscription avec votre binôme pour le projet 1
     dans le [document
     partagé](https://codimd.math.cnrs.fr/3f98v4-YT4CN2ktM6_g5lw?both).
     Inscrivez-vous aussi si vous n'avez pas encore de binôme!
- [ ] Téléchargez le sujet de TP `Semaine5` (rappel des
     [instructions](http://nicolas.thiery.name/Enseignement/IntroScienceDonnees/Devoirs/Semaine1/index.html#telechargement-et-depot-des-tps))
- [ ] Recopiez dans `Semaine5` vos feuilles de travail de la semaine 4:

        cd ~/IntroScienceDonnees/Semaine5
        cp ../Semaine4/?_*.md .

- [ ] Ouvrez la feuille [index](index.md) pour retrouver ces consignes.
- [ ] Consultez la section « Rapport » en fin de feuille.
- [ ] Révisez les [bonnes pratiques](../Semaine3/1_bonnes_pratiques.md)
	 vues en semaine 3.
- [ ] Partez à la découverte des [classificateurs](5_classificateurs.md).
- [ ] Reprenez votre analyse de la semaine dernière dans la feuille
     [analyse de donnees](4_analyse_de_donnees.md) avec vos propres
     classificateurs.


- [ ] Si nécessaire, relisez les
     [instructions](http://nicolas.thiery.name/Enseignement/IntroScienceDonnees/Students/Semaine1/index.html#telechargement-et-depot-des-tps)
     pour le téléchargement et le dépôt des TPs, ainsi que les 
	 [bonnes pratiques](http://nicolas.thiery.name/Enseignement/IntroScienceDonnees/Students/Semaine1/index.html#bonnes-pratiques).

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ec6c6b6156fb00ec8024441b2fabc1bf", "grade": false, "grade_id": "cell-4911a792a82448dE", "locked": true, "schema_version": 3, "solution": false}}

### Rapport

Cette feuille joue le rôle de mini-rapport ci-dessous. Elle vous
permettra à vous et votre enseignant ou enseignante d'évaluer rapidement votre
avancement sur cette première partie du projet. 

Au fur et à mesure, vous cocherez ci-dessus les actions que vous aurez
effectuées; pour cela, double-cliquez sur la cellule pour l'éditer, et
remplacez `- [ ]` par `- [x]`. Vous prendrez aussi des notes
ci-dessous. Enfin, vous consulterez la section « Revue de code »
ci-dessous pour vérifier la qualité de votre code.

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9de4c98836a55c9dbd666c730e4b7b3a", "grade": true, "grade_id": "cell-4911a792a82448dF", "locked": false, "points": 1, "schema_version": 3, "solution": true, "task": false}}

- [x] Vérifiez votre inscription avec votre binôme pour le projet 1
     dans le [document
     partagé](https://codimd.math.cnrs.fr/3f98v4-YT4CN2ktM6_g5lw?both).
     Inscrivez-vous aussi si vous n'avez pas encore de binôme!
- [x] Téléchargez le sujet de TP `Semaine5` (rappel des
     [instructions](http://nicolas.thiery.name/Enseignement/IntroScienceDonnees/Devoirs/Semaine1/index.html#telechargement-et-depot-des-tps))
- [x] Recopiez dans `Semaine5` vos feuilles de travail de la semaine 4:

        cd ~/IntroScienceDonnees/Semaine5
        cp ../Semaine4/?_*.md .

- [x] Ouvrez la feuille [index](index.md) pour retrouver ces consignes.
- [x] Consultez la section « Rapport » en fin de feuille.
- [x] Révisez les [bonnes pratiques](../Semaine3/1_bonnes_pratiques.md)
	 vues en semaine 3.
- [ ] Partez à la découverte des [classificateurs](5_classificateurs.md).
- [ ] Reprenez votre analyse de la semaine dernière dans la feuille
     [analyse de donnees](4_analyse_de_donnees.md) avec vos propres
     classificateurs.


- [ ] Si nécessaire, relisez les
     [instructions](http://nicolas.thiery.name/Enseignement/IntroScienceDonnees/Students/Semaine1/index.html#telechargement-et-depot-des-tps)
     pour le téléchargement et le dépôt des TPs, ainsi que les 
	 [bonnes pratiques](http://nicolas.thiery.name/Enseignement/IntroScienceDonnees/Students/Semaine1/index.html#bonnes-pratiques).

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "db64aca779ecc9f6030d7442c300f254", "grade": false, "grade_id": "cell-4911a792a82448dG", "locked": true, "schema_version": 3, "solution": false, "task": false}}

#### Revue du code

##### Affichage du code des principales fonctions

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 6744175f54caf5f72be76a7ffb88cc9d
  grade: false
  grade_id: cell-006fb180e6a86e73
  locked: true
  schema_version: 3
  solution: false
  task: false
---
from utilities import *
# Feuille 2_images.md
show_source(show_color_channels)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "abdfb5a24c36ae9a2cd982e3bcd4dba8", "grade": false, "grade_id": "cell-d21a1d7d40d56302", "locked": true, "schema_version": 3, "solution": false, "task": false}}

##### Conventions de codage

L'outil `flake8` permet de vérifier que votre code respecte les
conventions de codage usuelles de Python, telles que définies
notamment par le document [PEP
8](https://www.python.org/dev/peps/pep-0008/). Si la cellule suivante
affiche des avertissements, suivez les indications données pour
peaufiner votre code.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 698d7f2038bc259e761d6835254b4b13
  grade: true
  grade_id: cell-ce5cf24a9e83ff0e
  locked: true
  points: 2
  schema_version: 3
  solution: false
  task: false
---
assert run_without_error("flake8 utilities.py")
```

### Barême indicatif /20

* 1_tableaux.md : 1,5 points 
* 2_images.md : 2 points 
* 3_features.md : 1 point
* 4_analyse_de_données.md : 10 points 
* 5_classificateur.md : 4 points (+ 3 points bonus pour le OneR)
* index (semaine 4 et 5) : 1,5 points

```{code-cell} ipython3

```
