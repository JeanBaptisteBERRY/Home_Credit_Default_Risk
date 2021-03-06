# Home_Credit_Default_Risk

Beaucoup de personnes ont des difficultés à obtenir des prêts bancaires en raison de leurs antécédents en matière de crédit. Malheureusement, cette partie de la population est souvent pris pour cible par des prêteurs non fiables.

Home Credit se bat pour élargir l'inclusion bancaire pour les populations non bancarisées en fournissant une expérience positive, sûre et sécuritaire. Afin de s'assurer que cette population en difficultés ait une "expérience crédit" positive, Home Credit utilise une variété de données variées tels que telco (fournisseur de téléphonie) et des informations de transactions pour prédire les capacités de recouvrement de ses clients.

Alors qu'en ce moment, Home Credit utilise diverses méthode statistiques et des modèles de Machine Learning pour réaliser ces prédictions, ils défient des challengers pour les aider à débloquer de la valeur de leurs données. Agir ainsi permet de s'assurer que les clients capables de recouvrement ne sont pas rejetés et que les prêts sont accordés avec un remboursement (à échéances) adapté qui permettra à ces derniers de réussir dans la vie.

# Description des données

* **application_{train|test}.csv**
  * This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
  * Static data for all applications. One row represents one loan in our data sample.

* **bureau.csv**
  * All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
  * For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
  
* **bureau_balance.csv**
  * Monthly balances of previous credits in Credit Bureau.
  * This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

* **POS_CASH_balance.csv**
  * Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
  * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

* **credit_card_balance.csv**
  * Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
  * This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.
  
* **previous_application.csv**
  * All previous applications for Home Credit loans of clients who have loans in our sample.
  * There is one row for each previous application related to loans in our data sample.

* **installments_payments.csv**
  * Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
  * There is a) one row for every payment that was made plus b) one row each for missed payment.
  * One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

* **HomeCredit_columns_description.csv**
  * This file contains descriptions for the columns in the various data files.
  
 # Schéma Global :

![image](home_credit.png)

Les fichiers installments_payments, POS_CASH_BALANCE et previo_application sont à télécharger (trop lours pour être uploader sur github)  en allant sur le lien suivant : https://www.kaggle.com/c/home-credit-default-risk/data


