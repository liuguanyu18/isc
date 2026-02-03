---
name: field-semantics
description: Interpret column names and aliases (e.g., Pwr -> Power) to explain data semantics; use when a user needs field meaning analysis, column mapping, or a semantic report for load datasets.
---

# Field Semantics

- Match column names to a semantic dictionary (power, load, voltage, temperature, time).
- Resolve abbreviations and aliases with confidence scores.
- Output a field mapping table (original name, canonical name, meaning, confidence).
- Flag ambiguous columns for user clarification.
