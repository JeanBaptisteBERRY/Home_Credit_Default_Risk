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
* **POS_CASH_BALANCE:** Les données mensuelles concernant des précédents des terminaux de points de ventes (TPV) ou un prêt cash (cash loan en anglais aussi connu comme payday loan; il s'agit d'une forme de prêt de petit montant, à court terme, pas très fiable, avec intérêt élévé qui en général est remboursé au prochain salaire reçu par l'emprunteur) ont eu chez Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
* **credit_card_balance:** monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
* **installments_payment:** payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.
