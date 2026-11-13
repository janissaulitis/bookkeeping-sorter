"""Auto Books: sorts a month of bank transactions into accounting categories. Run: streamlit run app.py"""
import importlib

import pandas as pd
import streamlit as st

import kit
import sorter as S

S = importlib.reload(S)  # Streamlit Cloud keeps old copies of imported files after an update; always use the latest
kit = importlib.reload(kit)

kit.page("Auto Books", "Sorts your bank transactions for the accountant and learns from every fix", mark="📒",
         meta="For small companies and their accountants · no AI service, nothing leaves the app",
         badge="Free · no sign-up")

CATS = sorted(S.VAT)
st.markdown("""<style>
.bars { display: grid; gap: 7px; margin: 4px 0 8px; }
.bar { display: grid; grid-template-columns: minmax(120px, 34%) 1fr 72px; align-items: center; gap: 10px; font-size: 14px; }
.bar span { color: var(--k-ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar div { height: 12px; background: var(--k-line); border-radius: 6px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--k-go); border-radius: 6px; }
.bar b { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; color: var(--k-ink); }
</style>""", unsafe_allow_html=True)


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

    with st.container(border=True):
        if len(todo):
            st.markdown(f"### {done} of {len(s)} transactions sorted. {len(todo)} need you.")
            st.markdown(f"It left {'these' if len(todo) > 1 else 'this one'} because it isn't sure, usually a supplier it hasn't seen before. "
                        "Pick the right category below and it learns them.")
        else:
            st.markdown(f"### All {len(s)} transactions for {label} are sorted ✅")
            st.markdown("Ready for the accountant: download the file below.")
        m = st.columns(4)
        m[0].metric("Money in", money(income))
        m[1].metric("Business costs", money(spend), help="Without tax payments, loan repayments and owner transfers.")
        m[2].metric("VAT to pay, rough", money(vat["to_pay"]),
                    help=f"VAT in sales {money(vat['sales_vat'])} minus VAT you can reclaim on purchases "
                         f"{money(vat['reclaimable_vat'])}. Standard rates per category; the accountant confirms it.")
        m[3].metric("Left to check", len(todo))

    if len(todo):
        st.markdown("#### Needs you")
        ed = st.data_editor(
            todo[["date", "description", "amount", "suggested"]].rename(columns={"suggested": "category"}),
            hide_index=True, width="stretch", disabled=["date", "description", "amount"], key=f"ed_{month}",
            column_config={"category": st.column_config.SelectboxColumn("Category (best guess, change it)", options=CATS,
                                                                        required=True),
                           "amount": st.column_config.NumberColumn(format="%.2f")})
        if st.button(f"✓  Confirm these {len(todo)} and re-sort", type="primary"):
            for _, r in ed.iterrows():
                st.session_state.fixes[(r.description, r.amount)] = r.category
            st.rerun()

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Where the money went")
        biz = s[(s.amount < 0) & ~s.suggested.isin(["Owner transfers", "Loan repayment", "Taxes"])]
        costs = biz.groupby("suggested").amount.sum().abs().sort_values(ascending=False)
        top = costs.max() or 1
        st.markdown("<div class='bars'>" + "".join(
            f"<div class='bar'><span>{k}</span><div><i style='width:{v / top * 100:.1f}%'></i></div><b>{money(v)}</b></div>"
            for k, v in costs.items()) + "</div>", unsafe_allow_html=True)
        st.caption("Business costs only. Taxes, loan repayments and owner transfers are left out.")
    with right:
        st.markdown("#### What it learned this session")
        if st.session_state.fixes:
            st.dataframe(pd.DataFrame([{"Transaction": d, "Category": c} for (d, _), c in st.session_state.fixes.items()]),
                         hide_index=True, width="stretch", height=240)
            if st.button("Forget my fixes"):
                st.session_state.fixes = {}
                st.rerun()
        else:
            st.caption("Nothing yet. Every category you confirm is added to what it learns from, "
                       "so next month the same supplier is sorted automatically.")

    with st.expander(f"All {len(s)} transactions"):
        show = s[["date", "description", "amount", "suggested", "confidence", "status"]].rename(
            columns={"suggested": "category"})
        st.dataframe(show, hide_index=True, width="stretch", height=360,
                     column_config={"amount": st.column_config.NumberColumn(format="%.2f"),
                                    "confidence": st.column_config.ProgressColumn("sure", format="%.0f%%",
                                                                                  min_value=0, max_value=1)})
        st.download_button("Download for the accountant (CSV)", show.to_csv(index=False).encode("utf-8"),
                           file_name=f"sorted-{month}.csv", mime="text/csv", type="primary")

with tab_test:
    c = curve()
    first, last = c.iloc[0], c.iloc[-1]
    wrong = int(c.wrong_auto.sum())
    st.markdown(f"#### It sorted {first.auto:.0%} on its own in the first month and {last.auto:.0%} by the third, "
                f"with {wrong} mistakes in what it sorted alone.")
    st.markdown("The test copies real use. It learned from 9 months a bookkeeper had already sorted, then got the next "
                "3 months one at a time, which it had never seen. Those months bring in suppliers it didn't know "
                "(a new fuel station, Wolt, Ryanair, a new phone operator). After each month, the bookkeeper fixed what "
                "it left to check, and it learned from that before the next month.")
    m = st.columns(3)
    m[0].metric("Sorted alone, month 3", f"{last.auto:.0%}", f"{(last.auto - first.auto) * 100:+.0f} pts since month 1")
    m[1].metric("Right, when it sorted alone", f"{c.auto_right.min():.0%}")
    m[2].metric("Left to check, month 3", int(last.to_check), help="Out of " + str(int(last.rows)))
    st.table(pd.DataFrame({"Month": [pd.Timestamp(x + "-01").strftime("%B %Y") for x in c.month],
                           "Transactions": c.rows, "Sorted alone": [f"{x:.0%}" for x in c.auto],
                           "Left to check": c.to_check, "Sorted wrongly": c.wrong_auto}).set_index("Month"))
    st.caption(f"It only sorts on its own when it is at least {S.SURE:.0%} sure; everything else goes to a person. "
               "The data is an invented year for a small Latvian company (about 55 transactions a month) in mixed "
               "Latvian and English, written the way bank exports look. Real books will be messier, so measure on your "
               "own sorted months first.")

kit.footer("sorter.py · learns on your data, runs in the app")
