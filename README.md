# NHANES Age Prediction

Competition solution for the NHANES Age Prediction Hackathon (Summer Analytics 2026).

## Objective

Predict `age_group` (Adult / Senior) from NHANES tabular features. Evaluation metric: **F1 Score**.

## Project Structure

```
NHANES-Age-Prediction/
├── data/raw/           # Raw train, test, and sample submission CSVs
├── data/processed/     # Processed datasets (future phases)
├── src/                # Reusable Python modules
├── outputs/            # Validation and audit artifacts
├── logs/               # Application logs
├── models/             # Saved models (future phases)
├── notebooks/          # Final competition notebook (future phase)
└── submissions/        # Submission files (future phases)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place `Train_dataset.csv`, `Test_dataset.csv`, and `Sample_submission.csv` in `data/raw/`.

## Phase 1: Project Foundation

Run the dataset audit:

```bash
python src/project_setup.py
```

This validates schemas, targets, duplicates, and saves summary artifacts to `outputs/`.

## Target Mapping

| Label  | Encoded |
|--------|---------|
| Adult  | 0       |
| Senior | 1       |

## License

Private competition project.
