# Optimisation de Portefeuille sous Contraintes Pratiques (MILP & CSP)

Arthur Louis / Manon Giraudeau / Noam Gisclon

---

## Présentation du projet

Ce projet étudie les limites de l’optimisation de portefeuille classique de Markowitz,
qui ne prend pas en compte les contraintes pratiques rencontrées en gestion de
portefeuille réelle.

Nous formulons le problème comme un problème d’optimisation combinatoire et le
résolvons à l’aide de la **programmation linéaire mixte en nombres entiers (MILP)**
et de la **programmation par contraintes (CSP)**.

L’objectif est de comparer ces deux approches sur des données financières réelles,
tout en intégrant des contraintes réalistes telles que la cardinalité des actifs,
la diversification sectorielle et les coûts de transaction.

---

## Contraintes prises en compte

Les contraintes pratiques suivantes sont modélisées :

- **Contrainte de cardinalité** : nombre maximal d’actifs dans le portefeuille
- **Contraintes de diversification sectorielle**
- **Coûts de transaction et rééquilibrage du portefeuille**
- **Portefeuilles long-only**
- **Contrainte de budget**

Le risque est modélisé à l’aide de :
- l’optimisation moyenne–variance (Markowitz, baseline)
- extensions vers des mesures de risque de type **CVaR / drawdown**

---

## Technologies utilisées

- Python  
- Gurobi (optimisation MILP / MIQP)  
- OR-Tools (formulation CSP)  
- cvxpy (optimisation convexe)  
- pandas, numpy  
- yfinance (données financières)  
- matplotlib / plotly (visualisation)  
- pytest (tests automatisés)  

---

## Structure du dépôt
Gr02-Louis_Giraudeau_Gisclon-Sujet40/
├── README.md
├── src/
│ ├── main.py # Script principal d’exécution
│ ├── optimizer.py # Modèles d’optimisation (Markowitz, MILP, CSP)
│ ├── data_utils.py # Chargement et prétraitement des données financières
│ └── gui_steamlt_v2.py # Interface utilisateur (Streamlit)
├── docs/ # Documentation technique et théorique
├── slides/ # Support de présentation


