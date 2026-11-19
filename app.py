"""Auto Books: sorts a month of bank transactions into accounting categories. Run: streamlit run app.py"""
import os
import html
import importlib

import pandas as pd
import streamlit as st

import kit
import sorter as S

S = importlib.reload(S)  # Streamlit Cloud keeps old copies of imported files after an update; always use the latest
kit = importlib.reload(kit)

kit.page("Auto Books", "Sorts your bank transactions for the accountant and learns from every fix", mark="book", icon="📒",
         meta="For small companies and their accountants · no AI service, nothing leaves the app",
         badge="Free · no sign-up", theme="brutal")

kit.demo(os.environ.get("DEMO_VIDEO", "https://janis-saulitis-portfolio.netlify.app/media/books.mp4"), "Auto Books in 1 minute",
         "1. Use the **sample company** or upload your bank export.\n2. It sorts every transaction and leaves the ones it isn't sure about for you.\n3. Check its guesses and **confirm**: next month it sorts those suppliers by itself.\n4. **How well it sorts** shows the test on 3 months it never saw.")

CATS = sorted(S.VAT)


@st.cache_data(show_spinner=False)
def history():
    return S.load()


@st.cache_resource(show_spinner=False)
def model_for(upto: str, fixes: tuple):
    """Trained on every month before `upto` plus the corrections made in this session."""
    df = history()
    known = df[df.month < upto][["description", "amount", "category"]]
    if fixes:
        known = pd.concat([known, pd.DataFrame(list(fixes), columns=["description", "amount", "category"])])
    return S.train(known)


@st.cache_data(show_spinner=False)
def curve():
    return pd.DataFrame(S.learning_curve(history()))


def money(x):
    return f"€{x:,.0f}"


st.session_state.setdefault("fixes", {})  # (description, amount) -> category

tab_sort, tab_test = st.tabs(["Sort a month", "How well it sorts"])

with tab_sort:
    df = history()
    months = sorted(df.month.unique())[-3:]
    names = {m: pd.Timestamp(m + "-01").strftime("%B %Y") for m in months}
    a, b = st.columns([3, 2])
    src = a.segmented_control("Data", ["Sample company", "Upload your bank export"], default="Sample company",
                              label_visibility="collapsed")
    if src == "Upload your bank export":
        up = st.file_uploader("CSV with columns date, description, amount (money out negative)", type="csv")
        st.caption("Sorted with what the model learned from the sample company. The file stays in this session.")
        if not up:
            st.stop()
        try:
            tx = pd.read_csv(up)
            tx.columns = [c.strip().lower() for c in tx.columns]
            tx = tx[["date", "description", "amount"]].dropna()
            tx["amount"] = pd.to_numeric(tx["amount"].astype(str).str.replace(",", ".").str.replace(" ", ""))
        except Exception:
            st.error("Couldn't read that file. It needs the columns date, description and amount.")
            st.stop()
        month, label = "9999-99", "your file"
    else:
        month = b.selectbox("Month", months, index=len(months) - 1, format_func=names.get, label_visibility="collapsed")
        tx = df[df.month == month][["date", "description", "amount", "category"]].reset_index(drop=True)
        label = names[month]

    fixes = tuple(sorted((d, a_, c) for (d, a_), c in st.session_state.fixes.items()))
    s = S.sort(model_for(month, fixes), tx)
    for i, r in s.iterrows():  # the accountant's fixes always win
        if (r.description, r.amount) in st.session_state.fixes:
            s.loc[i, ["suggested", "status", "confidence"]] = [st.session_state.fixes[(r.description, r.amount)], "fixed", 1.0]

    todo = s[s.status == "to check"]
    done = len(s) - len(todo)
    vat = S.vat_summary(s.assign(status=s.status.replace("fixed", "sorted")))
    income = s.amount[s.amount > 0].sum()
    spend = -s.amount[(s.amount < 0) & ~s.suggested.isin(["Owner transfers", "Loan repayment", "Taxes"])].sum()

    tiles = [("Money in", money(income)), ("Business costs", money(spend)), ("VAT to pay, rough", money(vat["to_pay"])),
             ("Left to check", str(len(todo)))]
    if len(todo):
        kit.answer("Sorted for you", f"{html.escape(label[:1].upper() + label[1:].split(' ')[0])}: {done} of {len(s)} sorted. "
                   f"<mark>{len(todo)} need{'s' if len(todo) == 1 else ''} you.</mark>",
                   f"It left {'these' if len(todo) > 1 else 'this one'} because it isn't sure, usually a supplier it hasn't "
                   "seen before. Pick the right category below and it learns them.", tiles, raw=True)
    else:
        kit.answer("Sorted for you", f"All {len(s)} transactions are sorted",
                   "Ready for the accountant: download the file below. The VAT figure is a rough guide; "
                   "the accountant confirms it.", tiles)

    if len(todo):
        st.markdown("#### Needs you")
        st.caption("Its best guess is filled in. Change any that are wrong, then confirm.")
        picks = {}
        for i, r in todo.head(30).iterrows():
            with st.container(border=True):
                a, b, c = st.columns([5, 2, 4], vertical_alignment="center")
                a.markdown(f"<div class='k-row-desc'>{html.escape(r.description)}</div>"
                           f"<div class='k-row-sub'>{r.date} · {r.confidence:.0%} sure</div>", unsafe_allow_html=True)
                b.markdown(f"<div style='text-align:right;font-weight:800'>{r.amount:,.2f} €</div>", unsafe_allow_html=True)
                picks[i] = c.selectbox("Category", CATS, index=CATS.index(r.suggested), key=f"pick_{month}_{i}",
                                       label_visibility="collapsed")
        if len(todo) > 30:
            st.caption(f"Showing the first 30 of {len(todo)}.")
        if st.button(f"✓  Confirm {'these ' + str(len(picks)) if len(picks) > 1 else 'this one'} and re-sort", type="primary"):
            for i, cat in picks.items():
                st.session_state.fixes[(todo.loc[i, "description"], todo.loc[i, "amount"])] = cat
            st.rerun()
        st.caption("Next month it sorts these suppliers by itself.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Where the money went")
        biz = s[(s.amount < 0) & ~s.suggested.isin(["Owner transfers", "Loan repayment", "Taxes"])]
        costs = biz.groupby("suggested").amount.sum().abs().sort_values(ascending=False)
        kit.bars([{"label": k, "value": v, "text": money(v)} for k, v in costs.items()])
        st.caption("Business costs only. Taxes, loan repayments and owner transfers are left out.")
    with right:
        st.markdown("#### What it learned this session")
        if st.session_state.fixes:
            kit.table([{"t": d, "c": c} for (d, _), c in st.session_state.fixes.items()],
                      [kit.col("Transaction", "t"), kit.col("Category", "c", "chip", {})], max_height=300)
            if st.button("Forget my fixes"):
                st.session_state.fixes = {}
                st.rerun()
        else:
            st.caption("Nothing yet. Every category you confirm is added to what it learns from, "
                       "so next month the same supplier is sorted automatically.")

    with st.expander(f"All {len(s)} transactions"):
        show = s[["date", "description", "amount", "suggested", "confidence", "status"]].rename(
            columns={"suggested": "category"})
        kit.table([{"d": r.date, "t": r.description, "a": f"{r.amount:,.2f}", "c": r.category, "p": r.confidence,
                    "s": {"sorted": "Sorted", "to check": "To check", "fixed": "Fixed by you"}[r.status]}
                   for r in show.itertuples()],
                  [kit.col("Date", "d", "muted"), kit.col("Description", "t", "strong"), kit.col("Amount €", "a", "num"),
                   kit.col("Category", "c"), kit.col("How sure", "p", "meter"),
                   kit.col("Status", "s", "chip", {"Sorted": "good", "To check": "warn", "Fixed by you": "info"})],
                  max_height=420)
        st.download_button("Download for the accountant (CSV)", show.to_csv(index=False).encode("utf-8"),
                           file_name=f"sorted-{month}.csv", mime="text/csv", type="primary")

with tab_test:
    c = curve()
    first, last = c.iloc[0], c.iloc[-1]
    wrong = int(c.wrong_auto.sum())
    kit.answer("Tested on 3 months it never saw",
               f"It sorted {first.auto:.0%} on its own in month 1 and {last.auto:.0%} by month 3",
               f"With <b>{wrong} mistakes</b> in what it sorted alone. It learned from 9 months a bookkeeper had already "
               "sorted, then got the next 3 months one at a time, with suppliers it didn't know (a new fuel station, "
               "Wolt, Ryanair, a new phone operator). After each month the bookkeeper fixed what it left to check, and it "
               "learned from that before the next.",
               [("Sorted alone, month 3", f"{last.auto:.0%}"), ("Right when it sorted alone", f"{c.auto_right.min():.0%}"),
                ("Left to check, month 3", f"{int(last.to_check)} of {int(last.rows)}")])
    st.markdown("#### Month by month")
    kit.table([{"m": pd.Timestamp(x.month + "-01").strftime("%B %Y"), "n": int(x.rows), "a": x.auto,
                "t": int(x.to_check), "w": int(x.wrong_auto)} for x in c.itertuples()],
              [kit.col("Month", "m", "strong"), kit.col("Transactions", "n", "num"), kit.col("Sorted alone", "a", "meter"),
               kit.col("Left to check", "t", "num"), kit.col("Sorted wrongly", "w", "num")])
    st.caption(f"It only sorts on its own when it is at least {S.SURE:.0%} sure; everything else goes to a person. "
               "The data is an invented year for a small Latvian company (about 55 transactions a month) in mixed "
               "Latvian and English, written the way bank exports look. Real books will be messier, so measure on your "
               "own sorted months first.")

kit.footer("sorter.py · learns on your data, runs in the app")
