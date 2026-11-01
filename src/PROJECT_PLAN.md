# EE 311 Project: Energy Efficiency of Residential Buildings

**Team:** Oğuz Kağan Göçmez (single-person project, so you do every task) · **Course:** EE 311 Intro. to AI, Machine Learning and Data Science

**Files in this folder (`src/`)**

| File | What it is |
|---|---|
| `../articles/` | Papers: the reference article (Tsanas & Xifara, 2012) and four related papers that use the same dataset (see §4). Only links are in the repo (`articles/README.md`); the PDFs are kept locally |
| `data/ENB2012_data.csv` / `.xlsx` | The dataset (same data; the notebook uses the CSV) |
| `main.ipynb` | Code: data loading, exploration, model comparison, tuning (§6), tests on unseen shapes (§7), input importance (§8) |
| `PROJECT_PLAN.md` | This file: what needs to be done |
| `report/project_plan.tex` / `.pdf` | Project plan report draft (IEEE format, 2 pages) |

---

## 1. The project in short
Predict a building's **heating load** and **cooling load** from 8 design parameters (shape, areas, height,
orientation, glazing). Compare several machine-learning regression models, compare our results with the
reference article, and if possible propose an improvement. Then write three reports and give one presentation.

## 2. What must be handed in (all via LMS, late work is penalized)

| # | Deliverable | Due | Length | Must contain |
|---|---|---|---|---|
| 1 | **Project plan** | 2 weeks | 1–2 pages | Title & team · abstract (1–2 paragraphs) · schedule of subtasks covering **all 7 checklist steps** (approach, deadline, who is responsible; also time for literature search, reports and presentation) · **≥ 5 references** with full citations |
| 2 | **Midterm report** | 4 weeks | 3–5 pages | Title & team · revised abstract · what is done so far (one section per subtask, what you learnt) · remaining work + revised schedule + who does what · updated references |
| 3 | **Final report** | Last week of semester | 5–10 pages | The sections in §7 below |
| 4 | **Source code** | With final report | one `.zip` | The notebook, with markdown explanations (these can replace a README) |
| 5 | **Presentation** | Last week of semester | 10–15 min | Recorded (e.g. Zoom), speaker's face visible, **you present the whole talk yourself**, slides (e.g. PowerPoint) |

All reports use the **IEEE journal template** (Word, or LaTeX on Overleaf) and **IEEE citation style**.

## 3. The dataset
768 buildings simulated in Ecotect (Athens climate): 12 building shapes × 3 glazing areas × 5 glazing layouts
× 4 orientations (720), plus 12 shapes × 4 orientations with no glazing (48). All buildings have the same volume.

| Column | Name | Distinct values |
|---|---|---|
| X1 | Relative compactness | 12 |
| X2 | Surface area | 12 |
| X3 | Wall area | 7 |
| X4 | Roof area | 4 |
| X5 | Overall height | 2 |
| X6 | Orientation (coded 2–5) | 4 |
| X7 | Glazing area (0, 10, 25, 40 % of floor area) | 4 |
| X8 | Glazing area distribution (0–5) | 6 |
| **Y1** | **Heating load**: output 1 | 586 |
| **Y2** | **Cooling load**: output 2 | 636 |

So this is a **regression** problem with **two outputs**.

**Data check (done):** no missing values, no duplicate rows, and the CSV matches the Excel file.
- The CSV uses `;` as separator and `,` as decimal point. The notebook handles this.
  (The 527 empty rows that were at the end of the file have been removed.)
- One value was rounded in the CSV export (6.366 → 6.37). This is negligible.

## 4. The reference article
A. Tsanas and A. Xifara, "Accurate quantitative estimation of energy performance of residential buildings using
statistical machine learning tools," *Energy and Buildings*, vol. 49, pp. 560–567, 2012, doi: 10.1016/j.enbuild.2012.03.003.

**What they did:** histograms and scatter plots → Spearman correlation + mutual information → robust linear
regression (IRLS) vs. random forest (RF) → 10-fold cross-validation repeated 100 times → errors MAE, MSE, MRE →
RF variable importance (checked with a random "probe" feature).

**Their main findings:**
- RF is far better than linear regression. Heating load is easier to predict than cooling load.
- Relative compactness and surface area carry the same information (correlation −1).
- Orientation (X6) and glazing distribution (X8) are not significantly correlated with the outputs.
  (Correlation only: orientation still changes the cooling load of four two-storey shapes, see notebook §3.6.)
- By RF importance, glazing area (X7) is the most important input.

**Numbers to compare against (out-of-sample, mean ± std):**

| Error | IRLS heating | RF heating | IRLS cooling | RF cooling |
|---|---|---|---|---|
| MAE | 2.14 ± 0.24 | 0.51 ± 0.11 | 2.21 ± 0.28 | 1.42 ± 0.25 |
| MSE | 9.87 ± 2.41 | 1.03 ± 0.54 | 11.46 ± 3.63 | 6.59 ± 1.56 |
| MRE (%) | 10.09 ± 1.01 | 2.18 ± 0.64 | 9.41 ± 0.80 | 4.62 ± 0.70 |

### Papers that use the same dataset (for "Related Work")
Both are listed in `../articles/README.md` and cited in `report/project_plan.tex`.

**J. Žegklitz and P. Pošík**, "Symbolic regression algorithms with built-in linear regression," 2017, arXiv:1701.03641.
- Compares symbolic regression (methods that search for a readable formula: GPTIPS, EFS, FFX) with tuned linear
  regression, RF and SVR on 9 benchmarks. Our heating data is called "ENH", cooling "ENC".
- 100 random 70/30 train/test splits. Reports **median test RMSE**, not MAE (to compare, use RMSE ≈ √MSE).
- Their results on our data (median test RMSE):

| | RF | SVR | EFS | FFX | mGPTIPS | GPTIPS | LR |
|---|---|---|---|---|---|---|---|
| Heating | **0.51** | 0.67 | 0.55 | 1.05 | 1.72 | 2.54 | 2.93 |
| Cooling | 1.63 | **1.28** | 1.64 | 1.79 | 2.28 | 2.91 | 3.25 |

- Takeaway: the best formula method (EFS) nearly matches RF on heating while giving a readable equation.
  A **tuned SVR beats RF on cooling**, so SVR is worth tuning (§5, step 6). Our untuned RF gives roughly
  RMSE ≈ 0.47 (heating) and ≈ 1.62 (cooling), in line with theirs.

**M. T. Yazici, S. Basurra, and M. M. Gaber**, "Edge machine learning: Enabling smart Internet of Things applications,"
*Big Data Cogn. Comput.*, vol. 2, no. 3, Art. no. 26, 2018, doi: 10.3390/bdcc2030026.
- Tests whether RF, SVM and a neural network (MLP) can be trained and run on a Raspberry Pi. Measures accuracy
  (R² for regression), speed and power use on 10 UCI datasets. Our dataset is used for both regression and classification.
- Takeaway: RF was the most accurate and used the least total energy; predicting one sample took under 1 ms.
- Limitation: only averages over all 10 datasets are reported (in charts), with no numbers for our dataset.
  Cite it in Related Work, but it cannot go in a results table.

**M. Castelli, L. Trujillo, L. Vanneschi, and A. Popovič**, "Prediction of energy performance of residential buildings:
A genetic programming approach," *Energy Build.*, vol. 102, pp. 67–74, 2015, doi: 10.1016/j.enbuild.2015.05.013.
(Listed in `../articles/README.md`; already cited in `report/project_plan.tex`.)
- Evolves a formula with genetic programming (geometric semantic GP), adding a local search and linear scaling
  ("HYBRID-LIN"). Same protocol and errors as the reference article: 10-fold CV × 100, MAE / MSE / MRE (medians of 100 runs).
- Best test results (HYBRID-LIN): **MAE 0.51 (heating), 1.18 (cooling)**; MSE 0.47 / 3.33. Ties the article's RF on
  heating and beats it on cooling.
- **Errors in the paper, worth one sentence in the report:** Table 3 swaps the MSE and MRE rows for IRLS and RF
  (their "MRE" values are the article's MSE values and vice versa). Their heating MRE (0.62 %) is impossible:
  MRE ≥ 100 · MAE / largest heating load = 100 · 0.51 / 43.1 ≈ 1.2 %. So compare with this paper on **MAE only**.

**M. Sajjad, S. U. Khan, N. Khan, I. U. Haq, A. Ullah, M. Y. Lee, and S. W. Baik**, "Towards efficient building designing:
Heating and cooling load prediction via multi-output model," *Sensors*, vol. 20, no. 22, Art. no. 6419, 2020,
doi: 10.3390/s20226419. (Open access; listed in `../articles/README.md`.)
- Uses a recurrent neural network (GRU) on this tabular data, predicting the two loads separately and together
  (multi-output), after "preprocessing" (polynomial features + min–max normalization). Compares with SVR, RF, XGBoost,
  gradient boosting and an MLP, using an 80/20 hold-out split and 10-fold CV. States the load units as kWh/m².
- **Do not put its numbers in our results table.** Many rows report RMSE smaller than MAE, which is impossible
  (RMSE ≥ MAE always; e.g. Table 5, SVR heating: MAE 2.10, RMSE 1.28). The errors after preprocessing (e.g. GRU MAE 0.0102)
  appear to be on the normalized 0–1 scale, yet Table 7 compares them with other studies' errors in kWh/m².
- **Useful for Related Work:** Table 1 lists ~15 other studies on this dataset, and Table 7 lists their reported errors,
  including Chou & Bui (2014), whose PDF we do not have (so it is not cited in the plan report). Also relevant to idea 4 below (one model for both loads).

### Comparison with other work (MAE, random splits, i.e. known shapes)

| Study | Best method | How tested | Heating | Cooling |
|---|---|---|---|---|
| Tsanas & Xifara 2012 | Random forest | 10-fold CV × 100 | 0.51 | 1.42 |
| Castelli et al. 2015 | Genetic programming (HYBRID-LIN) | 10-fold CV × 100 (median) | 0.51 | 1.18 |
| **This project** | Tuned gradient boosting | 10-fold CV × 10, nested (§6.3) | **0.21** | **0.41** |

- Žegklitz & Pošík report RMSE (100 random 70/30 splits): 0.51 heating (RF), 1.28 cooling (SVR). Ours, roughly √MSE
  of tuned gradient boosting (§6.2): ≈ 0.32 and ≈ 0.57.
- Sajjad et al.'s Table 7 lists some studies with even lower errors (e.g. Nilashi et al.: 0.16 / 0.52; Roy et al.:
  0.037 / 0.127; Al-Rakhami et al., XGBoost: 0.175 / 0.307), with different and unchecked protocols. Check the
  originals before claiming that our result is the best; it is safest to say it beats the reference article and Castelli et al.
- **None of these papers tests on unseen building shapes.** Our §7 results (best MAE ≈ 3.1 heating, ≈ 3.3 cooling)
  have no published counterpart; this is our own contribution.

## 5. Step-by-step plan (the textbook's ML Project Checklist)

| Step | What to do | Status |
|---|---|---|
| 1. Frame the problem | Regression with 2 outputs. Metrics: MAE, MSE, MRE. Baseline to beat: the article. | ✅ Done (this file) |
| 2. Get the data | Load the CSV, drop the empty rows, give the columns readable names. | ✅ `main.ipynb` §2 |
| 3. Explore the data | Histograms, scatter plots, Spearman + mutual information, correlation heatmap (repeats the article's Fig. 1–2, Tables 2–4). | ✅ `main.ipynb` §3 |
| 4. Prepare the data | No cleaning needed. Scaling is done inside pipelines. **To try:** one-hot encode X6 and X8; drop the duplicate X2. | 🟡 Basic version done |
| 5. Shortlist models | 10-fold CV × 10 repeats on 6 models: Linear, Robust linear (Huber), k-NN, SVR, Random Forest, Gradient Boosting. | ✅ First pass, `main.ipynb` §5 |
| 6. Fine-tune | Hyperparameter search (`RandomizedSearchCV`, 40 settings per model) for RF, Gradient Boosting and SVR. Nested-CV check done (§6.3). **Still to do:** final run with `N_REPEATS = 100`. | 🟡 Search + check done, `main.ipynb` §6 |
| 7. Present results | Final tables and figures, comparison with the article and other papers, reports, presentation. | ⬜ To do |

**First results (MAE, 10-fold CV × 10, untuned models):**

| Model | Heating | Cooling |
|---|---|---|
| Random Forest | **0.31** | **0.98** |
| Gradient Boosting | 0.34 | 1.02 |
| SVR | 0.55 | 1.13 |
| k-NN | 1.42 | 1.66 |
| Robust linear (Huber) | 2.03 | 2.20 |
| Linear Regression | 2.09 | 2.27 |
| *Article: IRLS / RF* | *2.14 / 0.51* | *2.21 / 1.42* |

Our linear results are close to the article's IRLS, and our random forest already beats the article's.

**After tuning (MAE, 10-fold CV × 10, `main.ipynb` §6.2):**

| Model | Heating (untuned → tuned) | Heating, nested | Cooling (untuned → tuned) | Cooling, nested |
|---|---|---|---|---|
| Gradient Boosting | 0.34 → **0.20** | **0.21** | 1.02 → **0.36** | **0.41** |
| SVR | 0.55 → 0.34 | 0.34 | 1.13 → 0.87 | 0.87 |
| Random Forest | 0.31 → 0.32 | 0.31 | 0.98 → 0.94 | 0.95 |
| *Article: RF* | *0.51* | | *1.42* | |

Tuned gradient boosting is the best model for both loads. Random forest does not improve with tuning.
The nested-CV check (§6.3, where the search never sees the test fold) agrees within 0.01, except for gradient
boosting on cooling (0.41 vs. 0.36). **Quote the nested numbers for tuned models in the report**, since they are the fairer estimate.

**On unseen building shapes (MAE, leave-one-shape-out CV, `main.ipynb` §7):**

| Model | Heating (random CV → unseen shape) | Cooling (random CV → unseen shape) |
|---|---|---|
| k-NN | 1.42 → **3.46** | 1.66 → 3.93 |
| Robust linear (Huber) | 2.03 → 3.49 | 2.20 → **3.75** |
| Linear Regression | 2.09 → 3.51 | 2.27 → 3.93 |
| Random Forest | 0.31 → 4.26 | 0.98 → 4.95 |
| Gradient Boosting (tuned) | 0.20 → 5.26 | 0.36 → 4.79 |
| SVR (tuned) | 0.34 → 17.85 | 0.87 → 16.90 |

On a shape the model has not seen, the best error grows 10–17× (from 0.20 / 0.36 to 3.46 / 3.75). The flexible models
(trees, SVR) get 4–50× worse, the linear models and k-NN only 1.7–2.4×, so the simple models become the best. So the random-CV
numbers (ours and the article's) only hold for new orientations and glazing options of **known** shapes.
Some shapes are easy (0.71, 0.69) and some very hard (0.82, 0.79). On shape 0.74 the tree models fail because the
redundant inputs (compactness, surface area) put it with the two-storey shapes (§7.2).

**Tuned for unseen shapes (MAE, search with grouped CV, nested leave-one-shape-out CV, `main.ipynb` §7.3):**

| Model | Heating (§7.1 default → grouped-tuned) | Cooling (§7.1 default → grouped-tuned) | Settings chosen |
|---|---|---|---|
| SVR | 11.33 → **3.11** | 10.52 → 3.69 | C ≈ 5, gamma ≈ 0.02 (vs. C ≈ 2000 when tuned with random CV) |
| k-NN | 3.46 → 3.34 | 3.93 → **3.26** | 50 / 100 neighbours |
| Gradient Boosting | 4.06 → 3.92 | 4.47 → 4.71 | depth 2 |
| Random Forest | 4.26 → 4.17 | 4.95 → 4.50 | |

Tuning with grouped CV lowers the best unseen-shape error by about 10–13 % (heating 3.46 → 3.11, cooling 3.75 → 3.26).
SVR changes the most because the grouped search picks much simpler settings. The tree models gain little.
**Recommended model:** tuned gradient boosting for known shapes (§6); SVR or k-NN tuned with grouped CV for new shapes.

## 6. Ideas for your own contribution
The course asks you to "perhaps propose your own algorithms/improvements". Possible directions, most interesting first:
1. **Fair test on unseen building shapes.** Buildings that differ only in orientation mostly have very similar loads
   (except the cooling load of four two-storey shapes, notebook §3.6),
   so random CV puts near-copies in both training and test sets, which makes the errors look very low. Use
   `GroupKFold` grouped by building shape (the 12 values of X1) and see how much worse each model gets.
   ✅ **Done** (`main.ipynb` §7, leave-one-shape-out CV): the best error grows 10–17×, and the simple models (linear, k-NN) become the best.
   ✅ Tuning with grouped CV also done (§7.3): about 10–13 % lower error on new shapes; SVR and k-NN become the best.
2. **Explain the importance difference.** The article ranks glazing area first; our random forest does not.
   Compare permutation importance and add a random "probe" feature, as the article did.
3. **Handle redundant inputs.** Drop X2 (same as X1) and check whether the linear models become more stable.
   §7.2 adds a second reason: on unseen shape 0.74, tree models that split on X1/X2 fail. Test dropping X1 and X2
   for the tree models under grouped CV.
4. **Predict both loads with one model** (multi-output), since the two loads are strongly correlated (ρ ≈ 0.97).
5. **Encode X6 and X8 as categories** (one-hot) and check whether the linear models improve.

## 7. Final report outline (required by the course)
- **Abstract**
- **I. Introduction and Overview**
- **II. Related Work**: other papers that use this same dataset
- **III. Dataset Description and Pre-processing**: features, distributions, plots (histograms, correlation, box plots)
- **IV. Method**: the regressors, any feature selection, the error measures
- **V. Experimental Results**: how data is split (CV), how models are tuned, results table, comparison with other work
- **VI. Conclusion and Future Work**: which method was best, how to improve
- **Acknowledgement**: **required** statement on how AI tools were used (literature, code, writing, slides),
  and that the author reviewed and validated their suggestions
- **References**: all cited in the text, IEEE style

## 8. Things only you can do
- [x] Register the topic in the course's topic list (your name in the "Student Names" column), if not done yet.
- [x] Team: single-person project (decided).
- [x] Put real dates into the schedule: 7 project weeks, Week 1 = October 19, 2026, Week 7 = November 30, 2026
      (last week of the semester). Plan in week 2, midterm in week 4, final report and presentation in weeks 6–7.
- [ ] Chou & Bui (2014), doi: 10.1016/j.enbuild.2014.07.036: removed from `report/project_plan.tex` because it has not
      been read. Cite it again (e.g. in Related Work of the midterm report) only after reading the original paper; its numbers
      (MAE 0.24 heating, 0.89 cooling) are known only second-hand from Sajjad et al. (2020), Table 7.
- [x] ≥ 5 references: done (13 in `report/project_plan.tex`). To find more Related Work, good starting points: the article itself, its reference list (e.g. Breiman's random
      forests paper, Hastie et al. *Elements of Statistical Learning*), the UCI dataset page
      (https://archive.ics.uci.edu/dataset/242/energy+efficiency), and papers that cite the article
      (Google Scholar → "Cited by"). The papers that cite it are your "Related Work".
