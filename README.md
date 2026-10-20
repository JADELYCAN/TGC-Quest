# TCG Quest — experimental automatic card centering

Run with Python 3.10+:

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5050

The backend automatically detects a card quadrilateral, rectifies perspective, and estimates printed-frame border widths. Results depend on photo quality, printed borders, and detection correctness. The frame boundaries are **heuristics** and may be wrong. This is not a trained grading model and does **not** output a PSA grade. A defensible grading system needs a labeled dataset, calibrated photo capture, specialized defect detection, validation against PSA grades, and confidence/error reporting.
