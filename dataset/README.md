# Dataset: Synthea Synthetic Patient Data

Generated locally with [Synthea](https://synthetichealth.github.io/synthea/) — 674 synthetic patients (600 alive, 74 deceased), Massachusetts population, default modules.

Command used:
```bash
./run_synthea -p 600
```

## Contents

- `notes/` — 674 `.txt` files, one per patient, containing consolidated narrative clinical notes across all of that patient's encounters. Each note block includes Chief Complaint, History of Present Illness (prose mentioning conditions), Social History, Allergies, and a `# Medications` section listing free-text medication names. This is the **free-text input** for the extraction task.
- `csv/patients.csv` — patient demographics, keyed by `Id`.
- `csv/encounters.csv` — one row per clinical encounter, keyed by `Id`, linked to `PATIENT`.
- `csv/conditions.csv` — ground-truth diagnoses, columns: `START, STOP, PATIENT, ENCOUNTER, SYSTEM, CODE, DESCRIPTION`.
- `csv/medications.csv` — ground-truth medications, columns: `START, STOP, PATIENT, PAYER, ENCOUNTER, CODE, DESCRIPTION, BASE_COST, PAYER_COVERAGE, DISPENSES, TOTALCOST, REASONCODE, REASONDESCRIPTION`.

## How this becomes labeled data

For each patient/encounter, the corresponding rows in `conditions.csv` and `medications.csv` (filtered by `PATIENT` and/or `ENCOUNTER` id) are the **ground-truth labels** — the `DESCRIPTION` fields are exactly what the model should extract from the matching note text. No manual labeling needed since Synthea generated both the note and the structured record from the same underlying simulation.

Next step (Phase 1 data prep, see `../PLAN.md`): parse `notes/*.txt` into per-encounter note segments, join against `conditions.csv`/`medications.csv` on patient (and date/encounter where possible) to build `(note_text, {diagnoses: [...], medications: [...]})` pairs, then split into train/eval sets.

Note: `claims.csv`, `claims_transactions.csv`, `observations.csv`, `procedures.csv`, etc. were intentionally **not** copied here (large, not needed for this task). They still exist in the raw Synthea output at `../../ClinicalExtract/synthea/output/csv/` if needed later.
