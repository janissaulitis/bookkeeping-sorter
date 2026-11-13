# Auto Books

Sorts a month of bank transactions into accounting categories, gives a rough VAT figure, and learns from every correction. No AI service: a small text model trained on transactions an accountant already sorted, running inside the app.

**How well it works (invented data, honest test):** trained on 9 sorted months, then fed the next 3 months one at a time, including suppliers it had never seen. It sorted 86% on its own in the first month and 94% by the third, with 0 wrong in what it sorted alone. Anything below 70% confidence goes to a person instead of being guessed.

- `data/make_transactions.py` builds one year for a fictional Latvian company (mixed Latvian/English, bank-export style)
- `sorter.py` is the model (character n-grams of the description + direction and size of the amount → logistic regression), the VAT estimate and the month-by-month test
- `app.py` is the Streamlit app

Run: `pip install -r requirements.txt && streamlit run app.py`
