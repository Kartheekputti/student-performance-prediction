# Student Performance Prediction

ML app that predicts student performance (Low / Average / High) from attendance, study hours, GPA, assignment and internal scores.

## Run

```bash
pip install -r requirements.txt
python src/train.py
python -m streamlit run app.py
```

## Project

- `app.py` — Streamlit UI
- `src/` — preprocessing, training, prediction
- `data/student_data.csv` — training data
- `models/` — trained model
