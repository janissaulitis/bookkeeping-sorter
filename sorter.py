"""Sorts bank transactions into accounting categories and learns from every correction.

The model reads the description the way a person skims it: by pieces of words (so "CIRCLE K LATVIA 4412" and
"CIRCLE K LATVIA 9031" look alike), plus whether money came in or went out and roughly how much. It is trained on
transactions an accountant already sorted, tested on the latest months it never saw, and anything it is unsure about
is left for a person instead of being guessed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

DATA = Path(__file__).parent / "data" / "transactions.csv"
SURE = 0.70  # below this confidence a transaction goes to "to check"
VAT = {  # what an accountant would normally apply; the estimate is only as good as this table
    "Sales income": 0.21, "Rent": 0.21, "Utilities": 0.21, "Phone and internet": 0.21, "Software and subscriptions": 0.21,
    "Fuel and car": 0.21, "Office supplies": 0.21, "Travel": 0.12, "Meals and hospitality": 0.21, "Marketing": 0.21,
    "Accounting and legal": 0.21, "Insurance": 0.0, "Salaries": 0.0, "Taxes": 0.0, "Bank fees": 0.0,
    "Loan repayment": 0.0, "Owner transfers": 0.0,
}
# Input VAT that can't be reclaimed in Latvia (simplified): meals and hospitality
NOT_DEDUCTIBLE = {"Meals and hospitality"}


def _extra(df: pd.DataFrame) -> pd.DataFrame:
    a = df["amount"].astype(float)
    return pd.DataFrame({"direction": np.where(a > 0, "in", "out"),
                         "size": pd.cut(a.abs(), [-1, 2, 20, 100, 500, 1500, 1e12],
                                        labels=["tiny", "small", "medium", "large", "big", "huge"]).astype(str)})


def build() -> Pipeline:
    feats = ColumnTransformer([
        ("text", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), lowercase=True, min_df=1, sublinear_tf=True),
         "description"),
        ("meta", OneHotEncoder(handle_unknown="ignore"), ["direction", "size"]),
    ])
    prep = FunctionTransformer(lambda d: pd.concat([d[["description"]].reset_index(drop=True),
                                                    _extra(d).reset_index(drop=True)], axis=1))
    return Pipeline([("prep", prep), ("feats", feats), ("model", LogisticRegression(max_iter=3000, C=8.0))])


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["month"] = df["date"].str[:7]
    return df


def train(labelled: pd.DataFrame) -> Pipeline:
    m = build()
    m.fit(labelled[["description", "amount"]], labelled["category"])
    return m


def sort(model: Pipeline, tx: pd.DataFrame) -> pd.DataFrame:
    p = model.predict_proba(tx[["description", "amount"]])
    out = tx.copy()
    best = p.argmax(1)
    out["suggested"] = model.classes_[best]
    out["confidence"] = p.max(1)
    out["status"] = np.where(out["confidence"] >= SURE, "sorted", "to check")
    return out


def vat_summary(sorted_tx: pd.DataFrame, col: str = "suggested") -> dict:
    """Rough VAT picture for the month: VAT inside sales vs reclaimable VAT inside purchases."""
    t = sorted_tx[sorted_tx["status"] == "sorted"] if "status" in sorted_tx else sorted_tx
    rate = t[col].map(VAT).fillna(0.0)
    vat_in_amount = t["amount"].abs() * rate / (1 + rate)
    sales_vat = vat_in_amount[t[col] == "Sales income"].sum()
    purchase = (t["amount"] < 0) & (~t[col].isin(NOT_DEDUCTIBLE))
    reclaim = vat_in_amount[purchase].sum()
    return {"sales_vat": sales_vat, "reclaimable_vat": reclaim, "to_pay": sales_vat - reclaim}


def evaluate(df: pd.DataFrame, test_months: int = 3) -> dict:
    months = sorted(df["month"].unique())
    tr, te = df[df.month.isin(months[:-test_months])], df[df.month.isin(months[-test_months:])]
    m = train(tr)
    s = sort(m, te)
    sure = s[s.status == "sorted"]
    return {
        "test_rows": len(te), "train_rows": len(tr),
        "accuracy_all": (s.suggested == s.category).mean(),
        "share_sorted": len(sure) / len(s),
        "accuracy_sorted": (sure.suggested == sure.category).mean() if len(sure) else float("nan"),
        "errors": s[s.suggested != s.category][["date", "description", "amount", "category", "suggested", "confidence"]],
        "months": (months[-test_months], months[-1]),
    }


def learning_curve(df: pd.DataFrame, test_months: int = 3) -> list[dict]:
    """Month by month, the way it runs for real: sort the month, the accountant fixes what was left to check
    (and anything wrong), the model retrains on everything so far, then the next month comes in."""
    months = sorted(df["month"].unique())
    known = df[df.month.isin(months[:-test_months])]
    out = []
    for mo in months[-test_months:]:
        te = df[df.month == mo]
        s = sort(train(known), te)
        sure = s[s.status == "sorted"]
        out.append({"month": mo, "rows": len(s), "auto": len(sure) / len(s),
                    "auto_right": (sure.suggested == sure.category).mean() if len(sure) else float("nan"),
                    "to_check": int((s.status != "sorted").sum()),
                    "wrong_auto": int((sure.suggested != sure.category).sum())})
        known = pd.concat([known, te])  # corrected month joins the training data
    return out
