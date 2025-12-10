# HAJBI DOUAE
<img src="douae.jpeg" style="height:300px;margin-right:300px; float:left; border-radius:10px;"/>

Numéro d'étudiant : 22007751
Classe : CAC2
# COMPTE RENDU – PROJET DATA SCIENCE  
## Cycle de vie complet d’un projet de Machine Learning

---

## Introduction générale

Ce compte rendu présente une analyse approfondie d’un ensemble d’expérimentations en Data Science visant à illustrer le **cycle de vie complet d’un projet de Machine Learning**, depuis l’acquisition des données jusqu’à l’évaluation des modèles. L’objectif pédagogique est de dépasser une simple exécution de scripts pour adopter une **démarche analytique, méthodologique et interprétative**, conforme aux standards académiques et professionnels.

Les travaux s’appuient sur des jeux de données standards issus de **Scikit-learn** ainsi que sur un jeu de données médical téléchargé depuis **Kaggle via KaggleHub**, afin de simuler un contexte réel de projet data.

---

## 1. Contexte et typologie des problèmes traités

Les expérimentations couvrent les **quatre grandes familles du Machine Learning** :

- **Classification supervisée** (Régression Logistique, SVM)
- **Régression supervisée** (Régression Linéaire)
- **Apprentissage non supervisé** (Clustering KMeans)
- **Analyse comparative de modèles**

Chaque approche répond à un type spécifique de problématique décisionnelle : prédiction de classes, estimation de valeurs continues ou découverte de structures cachées dans les données.

---

## 2. Acquisition et compréhension des données

### 2.1 Jeux de données Scikit-learn

Trois datasets de référence sont utilisés :

### a) Iris Dataset
- 150 observations
- 4 variables explicatives
- 1 variable cible (3 espèces de fleurs)
- Problème : **classification multiclasse**

### b) Diabetes Dataset
- 442 observations
- 10 variables explicatives
- Variable cible continue
- Problème : **régression**

### c) Breast Cancer Wisconsin Dataset
- 569 observations
- 30 caractéristiques numériques extraites d’images médicales
- Variable cible binaire : bénin / malin
- Problème : **classification médicale critique**

Ces datasets sont propres et normalisés, ce qui permet de se concentrer sur la logique des modèles.

---

### 2.2 Données externes – KaggleHub

Un jeu de données médical supplémentaire est téléchargé depuis Kaggle :

**Breast Tissue Impedance Measurements**

```python
import kagglehub
path = kagglehub.dataset_download(
    "tarktunataalt/breast-tissue-impedance-measurements"
)
```
3. Classification supervisée – Régression Logistique (Iris)
3.1 Objectif

Prédire l’espèce d’une fleur à partir de mesures morphologiques (longueur et largeur des pétales et sépales).

3.2 Méthodologie

Séparation des données en features (X) et target (y)

Découpage train/test (80 % / 20 %)

Entraînement d’un modèle de régression logistique multiclasse

3.3 Résultats

Accuracy : 100 %

Precision, Recall et F1-score parfaits pour toutes les classes

Matrice de confusion sans erreur

3.4 Interprétation

Le dataset Iris est quasi linéairement séparable, ce qui explique les excellentes performances. Ce cas représente une situation pédagogique idéale, rarement rencontrée dans des projets réels.

4. Régression supervisée – Régression Linéaire (Diabète)
4.1 Objectif

Prédire l’évolution quantitative du diabète chez des patients à partir de variables cliniques.

4.2 Modèle utilisé

Régression Linéaire

Hypothèse de relation linéaire entre variables explicatives et cible

4.3 Indicateurs de performance

MSE ≈ 2900

R² ≈ 0,45

4.4 Analyse

Le modèle explique environ 45 % de la variance, ce qui indique une capacité prédictive modérée. Cela suggère que le phénomène étudié présente des relations non linéaires, difficilement capturables par un modèle linéaire simple.

5. Apprentissage non supervisé – Clustering KMeans (Iris)
5.1 Objectif

Identifier des groupes naturels dans les données sans utiliser la variable cible.

5.2 Algorithme

KMeans avec 3 clusters

Distance euclidienne

Données numériques normalisées implicitement

5.3 Résultats

Une séparation très nette de l’espèce Setosa

Chevauchement entre Versicolor et Virginica

5.4 Interprétation

Le clustering confirme l’existence d’une structure partielle dans les données. L’approche est exploratoire et ne vise pas la prédiction, mais l’analyse de similarités.

6. Classification avancée – Support Vector Machine (Breast Cancer)
6.1 Contexte métier

La détection du caractère malin ou bénin d’une tumeur est un problème critique, où les conséquences des erreurs sont asymétriques.

6.2 Modèle choisi

SVM avec noyau RBF

Capacité à modéliser des frontières complexes

Très performant en haute dimension

6.3 Resultat

Accuracy élevée (≈ 96 %)

6.4 Analyse critique

La SVM trouve une frontière de séparation optimale avec une forte capacité de généralisation. En contrepartie :

Le modèle est moins interprétable

Le réglage des hyperparamètres est crucial

Le coût computationnel est plus élevé

7. Méthodologie transversale
7.1 Séparation Train / Test
```python
train_test_split(test_size=0.2, random_state=42)
```
Cette étape garantit la généralisation du modèle et empêche l’évaluation optimiste des performances.

7.2 Choix des métriques

Classification : Accuracy, Precision, Recall, F1-score

Régression : MSE, R²

Clustering : visualisation et cohérence géométrique

8. Bonnes pratiques et limites

Les performances élevées ne garantissent pas l’utilité métier

Les jeux de données standards sont simplifiés

Un projet réel nécessite :

nettoyage avancé

traitement des déséquilibres

validation croisée

interprétabilité des modèles

Conclusion générale

Ce projet illustre clairement que la Data Science est un processus structuré et itératif, reposant autant sur la compréhension du problème que sur la maîtrise des outils algorithmiques. Les expérimentations menées montrent que chaque modèle possède des forces et des limites, et que le choix d’une méthode doit toujours être guidé par le contexte métier, la nature des données et les objectifs décisionnels.

Ce travail constitue une base solide pour aborder des projets plus complexes en intelligence artificielle et en analyse de données appliquée.
