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

## Project Structure

```text
tx_scripts/
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

## MLflow Evaluation Targets

Track the following metrics:

- accuracy
- macro F1
- weighted F1
- top-3 accuracy
- per-class precision and recall

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
