# Plaid Merchant Code Classifier

This project uses Plaid-style transaction data to build, evaluate, register, and serve a merchant-code classification model with MLflow.

The practical problem: Plaid provides strong transaction enrichment, but production systems often need their own merchant taxonomy for underwriting, expense automation, spend intelligence, risk rules, or customer-facing categorization. This project treats Plaid categories as weak supervision, maps them into a custom merchant-code taxonomy, trains a reproducible classifier, and compares model predictions against the Plaid-derived labels.

## Goal

Predict a custom `merchant_code` for each transaction using transaction-level fields such as:

- `name`
- `merchant_name`
- `amount`
- `payment_channel`
- `account_subtype`
- transaction date features
- optional location fields when available

The first label source will be Plaid's `personal_finance_category.primary` or `personal_finance_category.detailed`, mapped into a project-specific taxonomy.

## Why not use Plaid labels directly?

Plaid categories are useful, but they are not the same as card-network Merchant Category Codes (MCCs), and they may not match a company's internal taxonomy. Some transactions may also be missing merchant/category enrichment, or may arrive from non-Plaid sources such as CSV uploads, accounting systems, processors, or bank statements.

This model is therefore framed as custom merchant-code prediction, not as exact MCC reconstruction.

## Label Strategy: Weak Supervision

Hand-labeling every transaction is not required. Plaid attaches a `personal_finance_category` to every enriched transaction, which is remapped through `merchant_code_taxonomy.yml` to produce the training label automatically. This is a weak-supervision setup: the upstream label is approximate, the taxonomy mapping is the bridge to the project-specific schema, and the model learns to predict the project taxonomy directly from transaction features.

The real value of the model is not beating the Plaid-derived labels on the labeled set (where the mapping is largely deterministic from `merchant_name`), but generalizing to transactions where Plaid enrichment is missing or weak: CSV uploads, non-Plaid bank feeds, accounting exports, and ambiguous merchant strings.

This framing has a known leakage risk. If the taxonomy mapping is essentially "if `merchant_name = Starbucks` then `Coffee`," and the model sees `merchant_name` as a feature, training accuracy can approach 100% by memorizing the mapping. Evaluation therefore emphasizes a held-out set with merchants unseen during training, and the confusion matrix is read as the primary diagnostic.

## Project Structure

```text
plaid_mcc_classifier/
├── README.md
├── merchant_code_taxonomy.yml        # planned mapping from Plaid labels to custom codes
├── data/
│   ├── raw/                          # Plaid Sandbox responses or sample JSON, gitignored
│   └── processed/                    # training tables, gitignored
└── scripts/
    ├── fetch_sandbox_transactions.py # planned: call Plaid Sandbox
    ├── build_training_data.py        # planned: flatten transactions and assign weak labels
    ├── train_baseline.py             # planned: TF-IDF + logistic regression baseline
    ├── evaluate_model.py             # planned: reports and confusion matrices
    └── serve_example.py              # planned: example request to served MLflow model
```

## Modeling Plan

1. Collect transactions from Plaid Sandbox or Plaid's official sample custom-user JSON.
2. Flatten the transaction JSON into a tabular training dataset.
3. Map Plaid's category fields into a custom `merchant_code` taxonomy.
4. Train a baseline sklearn text classification pipeline.
5. Log params, metrics, model artifacts, and evaluation outputs to MLflow.
6. Register the best model in the MLflow Model Registry.
7. Serve the registered model locally with MLflow model serving.

## Baseline Model

The first model should be intentionally simple:

- Combine text fields such as `name` and `merchant_name` into one transaction text column.
- Convert that text into numeric features with `TfidfVectorizer`.
- Add lightweight structured fields such as amount buckets or payment channel.
- Train `LogisticRegression` as a multiclass classifier.

This is the classification equivalent of a clean pace baseline in the Strava project: simple enough to inspect, strong enough to beat rules, and easy to compare against more complex models later.

### TF-IDF and the Feature Matrix

`TfidfVectorizer` produces a standard feature matrix where rows are transactions and columns are vocabulary words. Cell `(i, j)` is the TF-IDF weight of word `j` in transaction `i`:

- **TF** (term frequency): count of word `j` in transaction `i`.
- **IDF** (inverse document frequency): `log(N / docs_containing_word_j)`. Rare merchants get a large multiplier, common boilerplate tokens (`POS`, `purchase`) get a small one.
- The product rewards tokens that are frequent in the current transaction but rare overall, which is what makes them discriminative.

A typical merchant-text vocabulary runs to tens of thousands of tokens, while each transaction contains only a handful of words. The matrix is therefore stored as a `scipy.sparse.csr_matrix`, which records only the nonzero cells as `(row, column, value)` triples. This keeps the representation tractable across millions of transactions and is natively supported by `LogisticRegression` and `XGBClassifier`.

### Combining Text and Structured Features

The transaction text matrix is concatenated with the structured features (amount, time-of-day, payment channel, account subtype) using `scipy.sparse.hstack`:

```python
from scipy.sparse import hstack

X_text = tfidf.fit_transform(df['merchant_text'])           # sparse, ~30k cols
X_struct = scaler.fit_transform(df[numeric_features])       # dense, ~10 cols
X_full = hstack([X_text, X_struct]).tocsr()                 # unified sparse matrix
```

Both `LogisticRegression` and tree-based classifiers accept the combined sparse matrix directly, so no dense conversion is required downstream.

### Why Logistic Regression for the Baseline

Logistic regression with L2 regularization is well matched to sparse, high-dimensional inputs: the optimization is well-conditioned, training is fast even at tens of thousands of features, and the learned coefficients are interpretable per class (which words push a transaction toward `Coffee` versus `Restaurants`). Tree-based models scan each feature for split points; on a TF-IDF matrix most features are zero for most rows, so a large fraction of that scan work yields no useful split. Logistic regression is the natural baseline; tree models become competitive after structured features dominate or after aggressive vocabulary pruning.

### Beyond the Baseline

`XGBClassifier` with `objective='multi:softprob'` is the natural next step once the baseline is wired up. `multi:softprob` returns the full probability matrix (one column per class), which is needed for top-k metrics and downstream confidence-aware routing; `multi:softmax` returns only the predicted class and is rarely the right choice for production scoring. The relevant objective families to recognize in XGBoost are:

- **Binary classification:** `binary:logistic`, `binary:hinge`.
- **Multiclass classification:** `multi:softmax`, `multi:softprob`.
- **Regression:** `reg:squarederror`, `reg:absoluteerror`, `reg:logistic`.
- **Ranking:** `rank:pairwise`, `rank:ndcg`, `rank:map`.
- **Survival and count:** `survival:cox`, `survival:aft`, `count:poisson`.

For the merchant-code problem, the head-to-head against logistic regression is what gets logged to MLflow; the choice depends on whether the structured features carry enough nonlinear interactions to justify the additional cost.

## MLflow Evaluation Targets

Track the following metrics:

- accuracy
- macro F1
- weighted F1
- top-3 accuracy
- per-class precision and recall

### How to Read These Metrics

- **Accuracy** is the fraction of predictions exactly right. It can be misleading under class imbalance: a model that always predicts the majority class can score high accuracy while being useless on the long tail. It is reported as a familiar headline number, not as the primary objective.
- **Macro F1** averages F1 across classes with equal weight per class, making it the right summary when rare merchant codes matter as much as common ones.
- **Weighted F1** averages F1 weighted by class support, so it reflects overall correctness while still penalizing precision and recall failures.
- **Top-k accuracy** measures whether the true label sits in the top `k` predicted classes. It is the relevant metric when the downstream surface is a suggestion list rather than a single auto-assignment.
- **Per-class precision and recall** expose where the model fails on specific taxonomy classes, which aggregate metrics hide.

### A Note on ROC-AUC vs PR-AUC

ROC curves plot True Positive Rate (recall) on the Y axis against False Positive Rate (FP / actual negatives) on the X axis; this is distinct from PR curves, which plot precision against recall. ROC-AUC is threshold-free and useful for model-to-model comparison, but it can overstate performance under heavy class imbalance because most negatives are easy to reject. PR-AUC is the more honest summary in that regime. For multiclass taxonomy work with balanced or moderately imbalanced classes, macro F1 plus the confusion matrix is generally more informative than ROC-AUC; ROC-AUC is computed one-vs-rest per class and averaged when reported at all.

### Confusion Matrix as a Taxonomy Diagnostic

The confusion matrix is an `N x N` grid where rows are true classes, columns are predicted classes, and cell `(i, j)` is the number of times true class `i` was predicted as class `j`. The diagonal counts correct predictions; off-diagonal cells expose the structure of the model's mistakes. For taxonomy design, the confusion matrix is the primary feedback signal:

- **Symmetric confusion** between two classes (A predicted as B and B predicted as A at similar rates) indicates the two classes are not cleanly separable from the available features and may need to be merged in the taxonomy.
- **Asymmetric confusion** (A predicted as B frequently, but not the reverse) suggests a systematic bias, often driven by uneven training support.
- **A row scattered across many columns** indicates a class that is poorly defined or under-discriminated by the current features.
- **A column dense across many rows** indicates a class the model uses as a catch-all under uncertainty, often an "Other" or generic category.

Iterating on the taxonomy in response to these patterns is part of the model lifecycle, not a sign of model failure.

Log the following artifacts:

- classification report
- confusion matrix
- taxonomy mapping
- sample predictions
- model signature and input example

## Serving Target

The serving milestone is a registered MLflow model that accepts transaction-like JSON and returns a predicted `merchant_code` plus class probabilities.

Example local serving command:

```bash
mlflow models serve -m "models:/plaid_merchant_code_classifier/Production" -p 5001
```
