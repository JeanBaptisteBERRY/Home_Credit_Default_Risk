# INTRODUCTION

Dans le notebook "NOM", nous étudierons dans un premier temps, la compétition Machine Learning "Home Credit Default Risk". L'objectif de cette compétition est d'utiliser des données issues d'historiques de demandes de prêts pour prédire si un client va être capable de recouvrir ou non son prêt bancaire. Le premiers élément que l'on déduit assez rapidement est le fait qu'il s'agit d'un tâche standard de classification de ML supervisé :
* **Supervisé :** les labels sont inclus dans les données d'entraînement et l'objectif est d'entraîner un modèle capable d'apprendre à prédire les labels à partir des features (variables; N.B : Features are the variables found in the given problem set that can strongly/sufficiently help us build an accurate predictive model.)
* **Classification :** Le label est une variable binaire : O (sera capable de recouvrir son prêt dans les temps) et 1 (aura des difficultés à recouvrir le prêt).

Avant de commencer il convient de bien comprendre la différence entre un prêt et un crédit :
#### Qu’est-ce qu’un prêt ?
* **Qui donne, qui reçoit ? :** Lorsque nous parlons de prêt, nous nous référons à une transaction financière dans laquelle une entité financière ou un particulier prête un montant fixe d'argent à un autre particulier.
* **La somme :** Tout dépend de l’accord trouvé entre le préteur et l’emprunteur. La condition première de cet accord est que le bénéficiaire d’un prêt s’acquitte de la totalité du montant emprunté. Aussi, l’argent prêté est réel, c’est-à-dire que cet argent est déjà disponible chez le prêteur.
* **La mise à disposition :** L’argent prêté est viré automatiquement sur le compte du client.
* **La durée :** La durée du prêt est définie en fonction de la durée de remboursement. Une fois le prêt remboursé dans sa totalité, le client doit refaire une demande de prêt.
* **Les intérêts :** La majoration par intérêts est générée en fonction d’une période donnée. Plus la durée de remboursement est longue, plus les intérêts sont élevés.
* **Le remboursement :** Il peut être effectué en une seule fois ou en plusieurs versements, d’une façon régulière et par le biais de quotas définis au préalable.
#### Qu’est-ce qu’un crédit ?
* **Qui donne, qui reçoit ? :** Le crédit ne peut être octroyé que par un organisme financier pour un particulier ou une entreprise.
* **La somme :** Tout comme le prêt, la somme est négociée entre les deux parties. Cependant, le client dispose d’un montant qu’il n’est pas obligé d’utiliser en totalité. L’argent en question n’est pas directement disponible physiquement. Il y a donc une création monétaire qui correspond concrètement à une impression de billets de banque.
* **La mise à disposition :** Le client peut retirer ce montant grâce à une carte de crédit (Il y a dans ce cas une ouverture d’un nouveau compte), ou avec une carte de débit habituelle (dans le cas d’une autorisation de découvert).
* **La durée :** Le crédit est à durée indéterminée. C’est-à-dire que si le client ne demande pas sont arrêt, il sera renouvelle chaque mois. Il y a donc une ligne de crédit ouverte.
* **Les intérêts :** Ceux-ci sont générés par paliers en fonction du montant du retrait. Par exemple, si lors d’un retrait de 50 euros, vos intérêts sont de 4 euros, à partir de 80 euros de retrait, les intérêts pourraient s’élever à 12 euros, et ainsi de suite.
* **Le remboursement :** Le remboursement est mensuel et automatique. Généralement le 1er du mois, il correspond à la somme utilisée avec les intérêts.

Traditionnellement, le crédit est utilisé par les entreprises ou les particuliers pour couvrir des périodes de manque de liquidité ou le financement de certaines activités commerciales alors que les prêts sont plutôt destinés à un financement plus commun dans le but d’acquérir certains biens de consommations comme un ordinateur, une voiture ou un bien immobilier par exempl

### Data

Les données sont fournies par Home Credit, un service dédié à fournir des prêts bancaires à une population non bancarisée. Réussir à prédire si un client sera capable de recouvrir ou aura des difficultés à recouvrir son prêt est un besoin commercial critique et Home Credit propose cette compétition pour voir quelles sortes de modèles la communauté ML peut mettre au point pour les accompagner et les aider dans leurs tâches.

On rappelle que les 7 différents types de données sont :

* **application_train/application_test:** Les principales données d'entraînement et de test avec des informations sur chaque prêt souscrit auprès d'Home Credit. Chaque prêt a sa propre ligne / donnée et est identifié grâce à la varaible SK_ID_CURR. Les données d'entraînement contienne la variable cible qui indique 0 si le prêt a été recouvert et 1 si le prêt ne l'a pas été.
* **bureau:** Les données concernant les crédit précedemment souscrits par le client dans d'autres institutions financières. Chaque crédit préccédemment souscrit a sa ligne dans le fichier bureau mais attention : chaque crédit précédemment souscrit a sa propre ligne dans 'bureau' mais un prêt dans 'application' peut avoir plusieurs crédits souscrits précédemment (dans le passé).
* **bureau_balance:** Les données mensuelles concernant les crédits souscrits précédemment. Chaque ligne correspond à la balance d'un mois d'un crédit précédent. On en déduit qu'un seul crédit précédemment souscrit peut avoir plusieurs lignes dans ce fichier, un pour chaque mois de la durée totale du crédit précédemment souscrit par le client.
* **previous_application:** les souscriptions précédentes pour prêt à Home Credit des clients qui ont un prêt dans 'application'. Chaque prêt en cours dans 'application' peut avoir plusieurs prêts souscrits dans le passé. Chaque prêt souscrit dans le passé correspond à une ligne et est identifié par la variable SK_ID_PREV.
* **POS_CASH_BALANCE:** Les données mensuelles concernant des précédentes transactions associées à des terminaux de points de ventes (TPV) ou à un prêt cash dans le passé (cash loan en anglais aussi connu comme payday loan; il s'agit d'une forme de prêt de petit montant, à court terme, pas très fiable, avec intérêt élévé qui en général est remboursé au prochain salaire reçu par l'emprunteur) ont eu chez Home Credit. Chaque ligne correspond à un mois d'une précédente transaction associée à un TPV et  un prêt dans 'application' peut avoir plusieurs noeuds / lignes dans 'POS_CASH_BALANCE'.
* **credit_card_balance:** Les données / la balance mensuelles associées à des cartes de crédit précédemment souscrites par des clients. Chaque noeud / ligne correspond à la balance d'un mois d'une carte de crédit et donc ,une seule carte de crédit peut avoir plusieurs noeuds / lignes.
* **installments_payment:** historique de paiement des précédents prêts bancaires chez Home Credit. Il y a un noeud pour chaque paiement effectué et une ligne pour chaque échéance ratée.

En outre, les définitions de toutes les colonnes nous sont founies(dans 'HomeCredit_columns_description.csv') et un exemple du fichier à soumettre.

Dans le notebook "NOM", nous utiliserons uniquement pour l'instant les données d'entraînement et les données de test. Maintenant, si on veut vraiment pouvoir réaliser des modèles précis, nous avons plutôt intérêt à utiliser toutes les données. Pour l'instant, nous nous concentrerons sur un seul fichier. Cela nous permettra d'établir un 'fil rouge' de base qu'on pourra par la suite améliorer. C'est mieux de prendre le problème petit bout par petit bout que de plonger directement dedans et d'être complètement perdu.

### Rappel : Metric: ROC AUC

Après avoir prix connaissance de toutes les données, lire 'HomeCredit_columns_description.csv' nous aide beaucoup. Il faut comprendre la métrique avec laquelle notre soumission va être évaluée. Dans notre cas, il s'agira d'une métrique de classification très commune, le Receiver Operating Characteristic Area Under the Curve (ROC AUC, parfois appelé également AUROC). 

Le Reciever Operating Characteristic (ROC) donne le taux de vrais positifs (fraction des positifs qui sont effectivement détectés) en fonction du taux de faux positifs (fraction des négatifs qui sont incorrectement détectés) : 

![image](ROC-curve.png)

Une courbe sur ce graphe (comme la courbe bleue ou rouge) représente un seul modèle de ML. Le seuil (N.B (en anglais) a treshold; All the positive values above the threshold will be “True Positives” and the negative values above the threshold will be “False Positives” as they are predicted incorrectly as positives.) vaut 0 dans le coin droite en haut et augmente jusqu'à valoir 1 au coin gauche en bas du graphique. Plus un modèle a une AUC (Area Under The Curve) élevée, plus le modèlee est précis / mieux. Par exemple, le modèle bleu est meilleur que le modèle rouge qui est meilleur que la diagonale noire qui indique un modèle aléatoire naïf. La diagonale divise l'espace du ROC : les points au-dessus de la diagonale représente des bons résultats de  classification (mieux que le hasard) et les points en dessous de la diagonale représente de mauvais résultats de  classification (pire que le hasard).

L'AUC est donc l'intégral de la courbe. Cette métrique est comprise entre 0 et 1. Plus l'AUC est proche de 1, plus le modèle est précis.
Un modèle qui devine au hasard a un ROC AUC de 0.5.

---Attention :---
When we measure a classifier according to the ROC AUC, we do not generation 0 or 1 predictions, but rather a probability between 0 and 1. This may be confusing because we usually like to think in terms of accuracy, but when we get into problems with inbalanced classes (we will see this is the case), accuracy is not the best metric. For example, if I wanted to build a model that could detect terrorists with 99.9999% accuracy, I would simply make a model that predicted every single person was not a terrorist. Clearly, this would not be effective (the recall would be zero) and we use more advanced metrics such as ROC AUC or the F1 score to more accurately reflect the performance of a classifier. A model with a high ROC AUC will also have a high accuracy, but the ROC AUC is a better representation of model performance.

Not that we know the background of the data we are using and the metric to maximize, let's get into exploring the data. In this notebook, as mentioned previously, we will stick to the main data sources and simple models which we can build upon in future work.





https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction











