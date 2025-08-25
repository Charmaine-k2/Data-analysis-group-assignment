# Q4 — Database Integration

This folder contains the deliverables for Question 4.

## Contents
- `src/db_pipeline.py` → database pipeline script
- `db/south_africa.db` → SQLite database
- `data/processed/` → output CSVs (`joined_inner.csv`, `joined_full.csv`, `joined_interpolated.csv`)
- `docs/` → screenshots (schema, tables, joins, execution)
- `requirements.txt` → dependencies

# How to Run
.venv\Scripts\activate
pip install -r requirements.txt
python src/db_pipeline.py
