# Energy Efficiency of Residential Buildings

Machine learning project for EE 311 (Intro. to AI, Machine Learning and Data Science).

The goal is to predict a building's **heating load** and **cooling load** from 8 design parameters: relative compactness, surface area, wall area, roof area, overall height, orientation, glazing area and glazing area distribution.

The project uses the [Energy Efficiency dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency) (768 simulated buildings) from Tsanas & Xifara (2012). It compares several regression models, tunes them, and checks the results against the reference paper. It also tests the models on unseen building shapes and measures how important each input is.

## Structure
- `src/main.ipynb`: data exploration, model comparison and tuning
- `src/data/`: dataset (CSV and Excel)
- `src/report/`: project plan report (IEEE format)
- `src/PROJECT_PLAN.md`: working plan, literature notes and results so far
- `articles/`: list of reference papers with links (PDFs not included)

## Data and license
The dataset is from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
and is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you use it, cite:

> A. Tsanas and A. Xifara, "Accurate quantitative estimation of energy performance of residential buildings using
> statistical machine learning tools," *Energy and Buildings*, vol. 49, pp. 560–567, 2012.

The code in this repository is under the MIT License (see `LICENSE`).
