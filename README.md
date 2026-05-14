<!-- Badges: build your brand at the top -->
[![GitHub stars](https://img.shields.io/github/stars/hemathens/kaggle-projects?style=social)](https://github.com/hemathens/kaggle-projects/stargazers)
[![Kaggle Profile](https://img.shields.io/badge/Kaggle-hem%20ajit%20patel-20BEFF?logo=kaggle)](https://www.kaggle.com/hemajitpatel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hem%20Ajit%20Patel-0A66C2?logo=linkedin)](https://www.linkedin.com/in/hem-patel19)
[![GitHub](https://img.shields.io/badge/GitHub-hemathens-181717?logo=github)](https://github.com/hemathens)

# Machine Learning Projects, Notebooks and Datasets

Welcome to my repository of Machine Learning projects, notebooks, and datasets.  
This repo showcases my journey and experiments in Machine Learning, Data Science, and Exploratory Data Analysis through real-world problems and custom-created datasets.

---

## Projects

| #  | IPYNBs                         | Description                                      | Link |
|----|----------------------------------|--------------------------------------------------|------|
| 1  | **Digits Prediction**            | Classify handwritten digits (MNIST).             | [![Digits](https://img.shields.io/badge/-Digits-7b61ff?style=for-the-badge&logo=python&logoColor=white)](https://www.kaggle.com/hemajitpatel/digits-prediction-hem) |
| 2  | **House Price Prediction**       | Predict house prices with regression models.     | [![HousePrice](https://img.shields.io/badge/-HousePrice-00b894?style=for-the-badge&logo=googlecloud&logoColor=white)](https://www.kaggle.com/hemajitpatel/house-price-hem) |
| 3  | **Titanic Survival**             | Who survived the Titanic? Feature engineering + models. | [![Titanic](https://img.shields.io/badge/-Titanic-0f4c81?style=for-the-badge&logo=gitlab&logoColor=white)](https://www.kaggle.com/hemajitpatel/titanic-hem) |
| 4  | **Heads or Tails**               | Predict heads or tail from a section of an image | [![HeadsOrTails](https://img.shields.io/badge/-HeadsOrTails-ff7b00?style=for-the-badge&logo=opencv&logoColor=white)](https://www.kaggle.com/code/hemajitpatel/heads-or-tails-hem) |
| 5  | **Superheros Abilities Dataset** | Sample usage notebook for superheroes dataset    | [![Superheroes](https://img.shields.io/badge/-Superheroes-1abc9c?style=for-the-badge&logo=superuser&logoColor=white)](https://www.kaggle.com/code/hemajitpatel/superheros-abilities) |
| 6  | **Rock vs Mine**                 | Predicts if an object is a rock or a mine using sonar data. | [![RvsM](https://img.shields.io/badge/-RvsM-ff3b30?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1yoUOlJD6ch8ZlxdqiLbBfI6iT6ozt-Al?usp=sharing) |
| 7  | **LLM Hallucination Evaluation** | Detect hallucinations in LLM responses with classical ML + leakage audit. | [![LLMHall](https://img.shields.io/badge/-LLMHallucination-6c3483?style=for-the-badge&logo=python&logoColor=white)](https://www.kaggle.com/code/hemajitpatel/llm-hallucination-evaluation) |

---

## 📓 Notebooks

---

### 1) 🔗 [Digits Prediction ](https://www.kaggle.com/hemajitpatel/digits-prediction-hem) 
**File:** `digits-prediction.ipynb`

**What’s inside**
- Dataset: MNIST (60k train / 10k test) — EDA showing sample digits and class balance.  
- Preprocessing: normalize pixel values to [0,1], reshape for CNN (`28x28x1`), one-hot encode labels.  
- Augmentation (optional): small rotations, shifts, random zoom to improve generalization.  
- Models included:
  - Baseline MLP (dense → relu → dropout → softmax).  
  - CNN (Conv2D → BatchNorm → ReLU → MaxPool → Dropout → Dense).  
  - Optionally a Transfer Learning variant using a small pretrained encoder (for experiments).
- Training artifacts: training/validation curves, confusion matrix, per-class accuracy, SavedModel export.

**How model is trained**
- Algorithm: Convolutional Neural Network (Keras/TensorFlow).  
- Loss / Optimizer: `categorical_crossentropy` with `Adam` (lr = `1e-3` default).  
- Batch size / Epochs: `batch_size=64`, `epochs=20–50` with `EarlyStopping(patience=5, restore_best_weights=True)`.  
- Callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `TensorBoard` for visual checks.  
- Regularization: Dropout (`0.25–0.5`), BatchNorm, L2 (optional).

**Data split**
- Use MNIST standard split: 60,000 training, 10,000 test.  
- Create validation from train (e.g., 90/10): → `train=54k`, `val=6k`, `test=10k`.  
- Example snippet:
```py
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=42)
```


### 2) 🔗 [House Price Prediction](https://www.kaggle.com/hemajitpatel/house-price-hem)  
**File:** `house-price.ipynb`

**What’s inside**
- Dataset: Kaggle House Prices — rich tabular dataset with numeric & categorical features.  
- EDA: distribution plots, missingness heatmap, correlations, target skew (log-transform).  
- Feature engineering: date features, polynomial & interaction terms, categorical encoding (One-Hot / Target Encoding), outlier handling.  
- Pipelines: `ColumnTransformer` for numeric + categorical, `Pipeline` for end-to-end training.  
- Models compared:
  - Linear Regression  
  - Lasso  
  - RandomForest  
  - XGBoost / LightGBM  
  - Stacking ensemble (meta-model).  
- Explainability: feature importance (tree SHAP), partial dependence plots.  

**How model is trained**
- Algorithm: Ensemble regression models (RandomForest, XGBoost, LightGBM) + stacking.  
- Loss / Objective: regression metrics (`MSE`, `RMSE`). Target often `log1p` transformed to stabilize variance.  
- Cross-validation: 5-fold CV with Out-of-Fold (OOF) predictions for stacking.  
- Hyperparameter tuning: `RandomizedSearchCV` or `Optuna` for RF, XGB, LGBM.  
- Typical tuning knobs:
  - XGBoost → `n_estimators=200–2000`, `learning_rate=0.01–0.2`, `max_depth=3–10`, `subsample=0.5–1.0`.  

**Pipeline sketch**
```py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
preprocessor = ColumnTransformer([...])
model = Pipeline([
    ('prep', preprocessor),
    ('clf', xgb.XGBRegressor(...))
])
```


### 3) 🔗 [Titanic Survival](https://www.kaggle.com/hemajitpatel/titanic-hem)  
**File:** `titanic.ipynb`

**What’s inside**
- EDA: survival rate by sex/class/age, missingness patterns (Age, Cabin).  
- Feature engineering: extract `Title` from Name, family size (`SibSp + Parch`), deck from Cabin (where possible), fill missing Age via median or model imputation.  
- Encoding: Sex → binary, Embarked → one-hot; Fare binned if useful.  
- Models compared:
  - Logistic Regression (baseline)  
  - RandomForest  
  - XGBoost  
- Model interpretability: coefficient table for Logistic Regression, SHAP or Permutation Importance for tree-based models.  

**How model is trained**
- Algorithms: `LogisticRegression` (L2), `RandomForestClassifier`, `XGBClassifier`.  
- Cross-validation: `StratifiedKFold(n_splits=5)` to preserve class balance.  
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC — plus confusion matrix.  
- Typical hyperparameters:
  - RandomForest → `n_estimators=100–500`, `max_depth=None` or tuned.  
  - XGBoost → `learning_rate=0.01–0.2`, `max_depth=3–8`.  

**Data split**
- Use stratified split to keep survival ratio consistent:  
```py
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```


### 4) 🔗 [Heads or Tails](https://www.kaggle.com/code/hemajitpatel/heads-or-tails-hem)  
**File:** `heads-or-tails.ipynb`

**What’s inside**
- Problem: classify an image patch as **heads** or **tails**. Dataset: custom / cropped images.  
- Preprocessing: resize (e.g., `128x128`), normalize, augmentation pipeline (flips, rotations, brightness jitter).  
- Model choices:
  - Lightweight custom CNN (baseline).  
  - Transfer learning using **MobileNetV2** or **EfficientNetB0** with fine-tuning for better accuracy.  
- Training utilities: class weights (if imbalance), `ImageDataGenerator` or `tf.data` pipeline, **Grad-CAM** for explainability.  

**How model is trained**
- Loss / Optimizer: `binary_crossentropy`, **Adam** or **SGD with momentum** for fine-tuning.  
- Batch size / Epochs: `batch_size=32`, `epochs=20–40`, with `EarlyStopping` + `ReduceLROnPlateau`.  
- Fine-tuning schedule: freeze base model for *N* epochs, then unfreeze top *k* layers and continue training with a lower learning rate (`1e-4 → 1e-5`).  

**Data split**
- Example: `train : val : test = 70 : 15 : 15`, or keep `train/val = 80/20` with a separate test set if available.  
- Ensure stratified split by label.  


### 5) 🔗 [Rock vs Mine (RvsM)](https://colab.research.google.com/drive/1yoUOlJD6ch8ZlxdqiLbBfI6iT6ozt-Al?usp=sharing)  
**File:** `rock-vs-mine.ipynb` (Colab link available)

**What’s inside**
- Dataset: Sonar / echo features (e.g., UCI Sonar dataset or similar).  
- Preprocessing: scaling with `StandardScaler`, optional signal preprocessing (smoothing, FFT features).  
- Models compared:
  - Logistic Regression (baseline)  
  - SVM (RBF, `probability=True`)  
  - RandomForest / XGBoost (strong tree baselines)  
- Model evaluation: Accuracy, ROC-AUC, confusion matrix, Precision, Recall — important for safety-critical detection.  

**How model is trained**
- Algorithms: SVM (RBF), RandomForest, XGBoost.  
- Cross-validation: `StratifiedKFold(n_splits=5)` with grid/random search.  
- Example hyperparameters:
  - SVM → `C ∈ {0.1, 1, 10}`, `gamma ∈ {'scale','auto'}` or tuned via logspace.  
  - XGBoost → `learning_rate=0.01–0.1`, `max_depth=3–6`.  
- Calibration / Tuning: decision threshold analysis using precision-recall tradeoff.  
  - If false negatives are costly, optimize for **recall** with constrained precision.  

**Data split**
- Typical:  
```py
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.10, stratify=y, random_state=1
)
```

> Notebooks live inside `/notebooks/` and include well-commented, reproducible code.

### 6) 🔗 [LLM Hallucination Evaluation](https://www.kaggle.com/code/hemajitpatel/llm-hallucination-evaluation)
**File:** `llm-hallucination-evaluation.ipynb`

**What's inside**
- Dataset: LLM Hallucination Benchmark — 200 annotated LLM responses across multiple domains, languages, and prompt types.
- EDA: binary label distribution, hallucination type breakdown (7 classes), domain spread, hallucination rate analysis.
- Leakage audit: systematic column-by-column ablation that identified 5 leaky post-labelling features (`severity`, `annotation_confidence`, `correction_text`, `hallucination_span`, `model_name`) that inflated scores to a perfect 1.00.
- Feature engineering:
  - Text: TF-IDF (200 features, unigrams, sublinear TF) on combined `prompt_text + [SEP] + response_text`.
  - Metadata: one-hot encoded `domain`, `language`, `prompt_type`, `task_type` + 5 length/ratio numeric features.
  - Combined into a single sparse matrix via `scipy.sparse.hstack`.
- Models included:
  - Logistic Regression at three regularisation strengths (C = 1.0, 0.1, 0.01).
  - XGBoost (shallow: `max_depth=2`, `n_estimators=50`) with class-weight correction.
- Evaluation: 5-fold stratified CV leaderboard with CV F1 (macro), CV AUC-ROC, Train F1, and overfit gap. Confusion matrix and per-domain F1 breakdown on the test set.

**How models are trained**
- Algorithms: Logistic Regression (`sklearn`) and XGBoost Classifier (`xgboost`).
- Class imbalance: handled via `class_weight='balanced'` (LR) and `scale_pos_weight=cw[1]/cw[0]` (XGBoost).
- Regularisation: C values swept over `{1.0, 0.1, 0.01}` to control overfitting on the small dataset; XGBoost capped at depth 2 and 50 trees.
- CV strategy: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` — stratification preserves the 65/35 class ratio in every fold.
- Best model (XGBoost shallow): CV F1 = `0.9742 ± 0.052`, CV AUC-ROC = `0.9984`.

**Data split**
- Stratified 70 / 15 / 15 train / val / test split on `label_binary`.
- At 200 rows: `train=140`, `val=30`, `test=30`.
- Example snippet:
```py
X_train, X_temp, y_train, y_temp = train_test_split(
    df, df['label_binary'],
    test_size=0.30, random_state=42, stratify=df['label_binary']
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50, random_state=42, stratify=y_temp
)
```
---

## 📁 Datasets

### 1. 🔗 [Superheroes Abilities Dataset](https://www.kaggle.com/datasets/hemajitpatel/superheros-abilities-dataset)
 
**File:** `superheros-abilities.ipynb`

**What’s inside**
- EDA: distribution of powers, missing data handling for free-text attributes.  
- Feature engineering: converting textual powers into categorical features (one-hot or embeddings), aggregate scores (e.g., `power_score`).  
- Experiments:
  - **Clustering**: KMeans / HDBSCAN to discover archetypes.  
  - **Classification / Regression**: predict alignment or power level using RandomForest / XGBoost.  
  - **Dimensionality reduction**: PCA / t-SNE / UMAP for visualizations.  

**How model is trained**
- Supervised tasks: RandomForest / XGBoost with 5-fold CV.  
- Clustering: hyperparameter sweep on `k` with silhouette scores / Davies-Bouldin index.  
- Text features: simple TF-IDF → PCA, or embedding pipeline if needed.  

**Data split**
- Predictive tasks: `train_test_split(X, y, test_size=0.20, random_state=42)` → 80/20.  
- Unsupervised tasks: use full dataset, with holdouts only for downstream validation.  

### 2. 🔗 [Code Similarity Dataset — Python Variants](https://www.kaggle.com/datasets/hemajitpatel/code-similarity-dataset-python-variants) 
**File:** `code-variants.ipynb`

**What’s inside**
- **Short summary**: many Python solutions for the same problems. Useful for finding similar code, detecting copies, and building a code search tool.  
- **Quick EDA**: count of problems, number of variants per problem, average lines and tokens.  
- **Preprocessing**: remove comments, normalize spacing, optionally rename variables to placeholders, and tokenize code.  
- **Prepared features**:
  - Token TF-IDF vectors for a fast baseline.  
  - Small AST-based counts (e.g., number of `if`, `for`, and `def` nodes).  
  - Optional pretrained code embeddings for stronger semantic matching.  
  - Handcrafted metrics such as cyclomatic complexity and number of function calls.  

**Experiments included**
- Fast baseline: TF-IDF on tokens + cosine similarity.  
- Pair classifier: embed two snippets and train a model to predict same/different problem.  
- Stronger route: fine-tune a pretrained code model for pairwise similarity.  
- Structural: convert AST into a simple graph and use a graph model.  
- Evaluation: retrieval metrics + ablations comparing token vs. structural approaches.  

**How the model is trained**
- **Two options**:
  1. Baseline: TF-IDF vectors + cosine similarity (no training).  
  2. Learned model: encoder maps snippets to vectors; trained so same-problem pairs are close, different-problem pairs are far.  
- **Typical settings**:
  - Batch size: small–medium depending on hardware.  
  - Optimizer: Adam or AdamW.  
  - Early stopping on validation performance.  
- **Practical tricks**:
  - Use balanced batches with equal positive and negative pairs.  
  - Periodically include hard negatives (look similar but are different).  
  - Use an ANN library for fast nearest-neighbor search at evaluation and deployment.  

**Simple training loop**
```py
for epoch in range(epochs):
    for code_a, code_b, label in dataloader:
        emb_a = encoder(code_a)
        emb_b = encoder(code_b)
        loss = contrastive_or_bce_loss(emb_a, emb_b, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
**Data split**

- Recommended split by problem so all variants of a problem go to the same partition.
- Example: 70% problems → train, 15% → validation, 15% → test.
- For pair training:
- Positive pairs: variants of the same problem.
- Negative pairs: different problems.
- 
Ensure test problems are unseen during training for realistic evaluation.

---

## 🔧 Tech Stack

## Tech Stack

| Tool / Library            | Purpose                                                                 |
|---------------------------|-------------------------------------------------------------------------|
| Python                    | Core programming language                                               |
| Pandas, NumPy             | Data manipulation & analysis                                            |
| Matplotlib, Seaborn       | Data visualization                                                      |
| Scikit-learn              | ML modeling, TF-IDF (TfidfVectorizer), pipelines, metrics               |
| TensorFlow / PyTorch      | Deep learning (future work)                                             |
| Jupyter Notebook          | Interactive code & documentation                                        |
| Kaggle API                | Dataset handling automation                                             |
| Git & GitHub              | Version control and collaboration                                       |
| NLTK / spaCy              | Tokenization, lemmatization, English stopwords lists                    |
| `TfidfVectorizer` (sklearn) | TF-IDF extraction — supports `stop_words='english'` or custom lists, n-grams, `min_df`, `max_df` |
| Custom stopwords (domain) | Add domain-specific tokens (e.g., variable names, common words) to stoplist |
| Sentence-Transformers / Transformers | Semantic embeddings for stronger text similarity / classification  |
| Gensim                    | Word2Vec / Doc2Vec / fast text embeddings                               |
| Hugging Face Tokenizers   | Fast tokenization for transformer models                                |
| imbalanced-learn (SMOTE)  | Oversampling for imbalanced classes                                      |
| Optuna / RandomizedSearch | Hyperparameter tuning (efficient search for model + vectorizer params)   |
| FAISS / Annoy             | Fast nearest-neighbor search for retrieval tasks                        |

---

## How to Use

### 1) Clone the Repo
```bash
# Clone the repository
git clone https://github.com/hemathens/kaggle-projects.git
cd kaggle-projects

# View datasets
cd datasets/
# View notebooks
cd notebooks/
```

### 2) Quick view (no install)

View notebooks on GitHub by clicking the .ipynb files — GitHub renders them read-only.

Or use nbviewer: https://nbviewer.org/github/hemathens/kaggle-projects/blob/main/path/to/notebook.ipynb

Or open in Colab (runs in browser):
```
https://colab.research.google.com/github/hemathens/kaggle-projects/blob/main/notebooks/your_notebook.ipynb
```
Replace notebooks/your_notebook.ipynb with the real path.

### 3) Run locally (recommended for full control)

**A. Create & activate a virtual environment**

Linux / macOS (bash/zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Windows (CMD):
```
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**B. Install dependencies**

If the repo includes requirements.txt:
```
pip install -r requirements.txt
```
If the repo includes environment.yml (conda):
```
conda env create -f environment.yml
conda activate <env-name>
```

**C. Optional: make the venv available as a Jupyter kernel**
```
pip install ipykernel
python -m ipykernel install --user --name=kaggle-projects-env --display-name "kaggle-projects-env"
```
Then pick this kernel inside Jupyter/Lab when opening notebooks.

**D. Start JupyterLab (recommended) or Notebook**
```
jupyter lab
# or
jupyter notebook
```
Open the notebook file (for example notebooks/digits-prediction.ipynb) in the browser and run cells.

**E. Kaggle dataset (if needed)**

Install Kaggle CLI:
```
pip install kaggle
```
Place kaggle.json (your API token) in:
-Linux/macOS: ~/.kaggle/kaggle.json (set permissions chmod 600 ~/.kaggle/kaggle.json)
-Windows: %USERPROFILE%\.kaggle\kaggle.json
Download dataset (example):
```
kaggle datasets download -d hemajitpatel/code-similarity-dataset-python-variants -p datasets/ --unzip
```

---

## ⚖️ Licences

These datasets are published under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

You are free to:

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material for any purpose

Under the following terms:

- **Attribution** — You must give appropriate credit by linking to [my Kaggle profile](https://www.kaggle.com/hemajitpatel), provide a link to the license, and indicate if changes were made.

Full License: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)
