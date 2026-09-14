# SION Visual Field Progression Web App

A small Streamlit application for longitudinal visual-field analysis using the 156-archetype SWIN framework.

## Expected CSV columns

Required:
- `PatID`
- `Eye`
- `Age`
- `MD`
- `AT1` ... `AT156`

For sensitivity maps:
- `Sens_1` ... `Sens_54`, with blind spots 26 and 35 omitted (52 columns)

For Bebie curves:
- `TD_1` ... `TD_54`, with blind spots 26 and 35 omitted (52 columns)

The app also accepts `Sens1`/`TD1` naming without underscores.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local address shown by Streamlit (usually `http://localhost:8501`).

## SION settings used

- 156 archetypes grouped as Normal / Superior / Inferior / Whole-field
- Top-K regional burden: K = 5
- Baseline: median of the first 3 VFs
- Persistence: 2 consecutive VFs
- User-adjustable thresholds:
  - superior presence
  - inferior presence
  - other-field presence
  - onset delta
  - worsening delta
- Dynamic regional reference is updated after a confirmed change

The five default threshold values in the interface are placeholders (`0.10`) and should be changed to the thresholds you want to evaluate.
