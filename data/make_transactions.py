"""Generates data/transactions.csv: one year of invented bank transactions for a small Latvian company,
each with the accounting category an accountant gave it.

Descriptions look like real bank exports: card terminals with branch numbers, invoice numbers, mixed Latvian and
English, the same supplier written differently from month to month. All companies and people are fictional,
except well-known utilities and chains, which appear only as the kind of text a statement would show.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

rnd = random.Random(11)

# category -> (VAT rate an accountant would apply, list of (description templates, amount range, times per month))
SPEC = {
    "Sales income": (0.21, [("Maksājums par rēķinu {inv} {client}", (300, 4200), 9),
                            ("Payment invoice {inv} {client}", (300, 4200), 4)]),
    "Rent": (0.21, [("SIA Biroju Centrs nomas maksa {mon}", (850, 850), 1)]),
    "Utilities": (0.21, [("Elektrum rēķins {num}", (60, 140), 1), ("Rīgas ūdens {num}", (18, 35), 1)]),
    "Phone and internet": (0.21, [("LMT rēķins {mon}", (35, 70), 1), ("Tet internets {num}", (25, 25), 1)]),
    "Software and subscriptions": (0.21, [("GOOGLE *Workspace {code}", (36, 54), 1), ("Microsoft*365 {code}", (22, 22), 1),
                                          ("ADOBE *CREATIVE CLD", (60, 60), 1), ("Zoom.us {code}", (14, 14), 1),
                                          ("CANVA* {code}", (12, 12), 1)]),
    "Fuel and car": (0.21, [("CIRCLE K LATVIA {term} RIGA", (35, 90), 4), ("VIADA DUS {term}", (30, 80), 2),
                            ("Neste {term} Jelgava", (35, 85), 1), ("Autoserviss Pārdaugava SIA", (80, 420), 0.25)]),
    "Office supplies": (0.21, [("Officeday.lv pasūtījums {num}", (20, 160), 1), ("DEPO DIY {term}", (15, 90), 0.5)]),
    "Travel": (0.12, [("airBaltic {code}", (90, 380), 0.4), ("Hotel Tallinn Old Town {num}", (85, 260), 0.3),
                      ("Bolt ride {code}", (5, 18), 2)]),
    "Meals and hospitality": (0.21, [("RESTORANS LIDO {term}", (25, 140), 1.5), ("Kafejnīca Kūkotava {term}", (8, 45), 1)]),
    "Marketing": (0.21, [("FACEBK *ADS {code}", (60, 400), 1), ("GOOGLE *ADS{num}", (80, 500), 1),
                         ("SIA Drukas Nams reklāmas materiāli", (90, 600), 0.3)]),
    "Accounting and legal": (0.21, [("Grāmatvedības pakalpojumi SIA Skaitļi {mon}", (220, 220), 1),
                                    ("Zvērinātu advokātu birojs Lex SIA", (300, 900), 0.15)]),
    "Insurance": (0.0, [("BTA apdrošināšana polise {num}", (45, 45), 1)]),
    "Salaries": (0.0, [("Alga {person} {mon}", (900, 2100), 3), ("Darba alga {person}", (900, 2100), 0.2)]),
    "Taxes": (0.0, [("VID PVN maksājums {mon}", (400, 1800), 1), ("VID VSAOI un IIN {mon}", (1100, 1900), 1)]),
    "Bank fees": (0.0, [("Komisija par konta apkalpošanu", (6, 6), 1), ("Komisijas maksa par maksājumu", (0.35, 1.5), 3)]),
    "Loan repayment": (0.0, [("Kredīta atmaksa līgums {num}", (420, 420), 1)]),
    "Owner transfers": (0.0, [("Pārskaitījums uz savu kontu", (200, 1500), 0.5), ("Dividenžu izmaksa", (1000, 3000), 0.15)]),
}
# Suppliers that only show up in the last three months, so the test includes names the model has never seen,
# the way a real company keeps adding new ones.
NEW = {
    "Sales income": [("Ienākošais maksājums {client} rēķins {inv}", (400, 3000), 2)],
    "Fuel and car": [("KOOL LATVIJA {term}", (30, 80), 2), ("Virši-A DUS {term}", (30, 80), 1)],
    "Software and subscriptions": [("NOTION LABS {code}", (10, 10), 1), ("OPENAI *CHATGPT SUBSCR", (22, 22), 1)],
    "Office supplies": [("Rimi Hiper biroja preces {term}", (12, 60), 0.6)],
    "Meals and hospitality": [("Lido Atpūtas centrs {term}", (30, 120), 0.7), ("Wolt order {code}", (15, 60), 1)],
    "Travel": [("Ryanair {code}", (40, 160), 0.5), ("BOLT.EU/O/{num}", (6, 20), 1)],
    "Marketing": [("LINKEDIN ADS {code}", (100, 300), 0.7)],
    "Phone and internet": [("Bite Latvija rēķins {num}", (20, 40), 1)],
    "Bank fees": [("Komisija SEPA zibmaksājums", (0.3, 0.6), 2)],
}
CLIENTS = ["SIA Ziemeļu Būve", "Baltic Retail Group", "SIA Mežaparka Klīnika", "Nordic Freight AS", "SIA Kurzemes Maiznīca",
           "Tallinn Digital OÜ", "SIA Daugavas Viesnīca"]
PEOPLE = ["Anna Kalniņa", "Mārtiņš Ozols", "Ilze Bērziņa", "A. Kalnina", "M. Ozols"]
MONTHS_LV = ["janvāris", "februāris", "marts", "aprīlis", "maijs", "jūnijs", "jūlijs", "augusts", "septembris",
             "oktobris", "novembris", "decembris"]


def fill(t, d):
    return t.format(inv=f"{rnd.choice(['', 'Nr.', 'INV-'])}{rnd.randint(1000, 9999)}", client=rnd.choice(CLIENTS),
                    mon=rnd.choice([MONTHS_LV[d.month - 1], f"{d.month:02d}/{d.year % 100}", f"{d.year}-{d.month:02d}"]),
                    num=rnd.randint(100000, 999999), code=f"{rnd.randint(10, 99)}{rnd.choice('ABCDXYZ')}{rnd.randint(100, 999)}",
                    term=rnd.randint(1000, 9999), person=rnd.choice(PEOPLE))


def main():
    rows = []
    for m in range(1, 13):
        first = date(2026, m, 1) if m <= 10 else date(2025, m, 1)
        for cat, (vat, templates) in SPEC.items():
            for tmpl, (lo, hi), per_month in templates:
                n = int(per_month) + (1 if rnd.random() < per_month % 1 else 0)
                for _ in range(n):
                    d = first + timedelta(days=rnd.randint(0, 27))
                    amt = round(rnd.uniform(lo, hi), 2)
                    sign = 1 if cat == "Sales income" else -1
                    rows.append({"date": d.isoformat(), "description": fill(tmpl, d), "amount": sign * amt,
                                 "category": cat, "vat_rate": vat})
        if first >= date(2026, 8, 1):
            for cat, templates in NEW.items():
                for tmpl, (lo, hi), per_month in templates:
                    n = int(per_month) + (1 if rnd.random() < per_month % 1 else 0)
                    for _ in range(n):
                        d = first + timedelta(days=rnd.randint(0, 27))
                        sign = 1 if cat == "Sales income" else -1
                        rows.append({"date": d.isoformat(), "description": fill(tmpl, d),
                                     "amount": sign * round(rnd.uniform(lo, hi), 2), "category": cat,
                                     "vat_rate": SPEC[cat][0], "new_supplier": True})
    df = pd.DataFrame(rows)
    df["new_supplier"] = df.get("new_supplier", False).fillna(False).astype(bool)
    df = df.sort_values("date").reset_index(drop=True)
    out = Path(__file__).with_name("transactions.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} transactions, {df.category.nunique()} categories -> {out}")


if __name__ == "__main__":
    main()
