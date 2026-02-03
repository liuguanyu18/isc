---
name: data-ingestion
description: Ingest and normalize campus load data from CSV/Excel/SQL sources; use when a user asks to connect data sources, detect schemas, or consolidate raw data into a consistent table for forecasting.
---

# Data Ingestion

- Detect the input source type (CSV, Excel, SQL connection) and confirm connection parameters.
- Load data into a tabular structure with explicit column types.
- Normalize time columns into a standard timestamp format.
- Emit a schema summary (column name, type, sample values).
- Return a normalized dataset handle for downstream skills.
