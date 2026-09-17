"""Model training jobs (scikit-learn and PyTorch).

Both jobs share training/data.py (identical time-based split + row
counts), write versioned artifacts via training/artifacts.py and register
runs in the Postgres model_runs table.
"""
