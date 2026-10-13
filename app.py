import joblib
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    team_stats = joblib.load("team_stats.joblib")
    h2h_table = joblib.load("h2h_table.joblib")
    meta = joblib.load("meta.joblib")
    return model, team_stats, h2h_table, meta


model, team_stats, h2h_table, meta = load_artifacts()
FEATURE_COLS = meta["feature_cols"]
TEAMS = meta["teams"]
CLASS_ORDER = meta["class_order"]
THRESHOLD = meta["confidence_threshold"]


def get_h2h(team, opponent):
    try:
        rec = h2h_table.loc[(team, opponent)]
        return float(rec["wl"]), float(rec["goals"])
    except KeyError:
        return 0.0, 0.0


def build_features(team_a, team_b):
    a = team_stats.loc[team_a]
    b = team_stats.loc[team_b]
    a_wl, a_goals = get_h2h(team_a, team_b)
    b_wl, b_goals = get_h2h(team_b, team_a)
    feat = {
        "appearances_diff": a["appearances"] - b["appearances"],
        "titles_diff": a["titles"] - b["titles"],
        "finals_diff": a["finals"] - b["finals"],
        "semifinals_diff": a["semifinals"] - b["semifinals"],
        "fifa_rank_diff": a["fifa_rank"] - b["fifa_rank"],
        "h2h_wl_diff": a_wl - b_wl,
        "h2h_goals_diff": a_goals - b_goals,
    }
    return pd.DataFrame([feat])[FEATURE_COLS]


st.set_page_config(page_title="World Cup 2018 Predictor", page_icon="⚽", layout="centered")

st.title("ระบบทำนายผลฟุตบอลโลก 2018 รอบแบ่งกลุ่ม")
st.caption("ระบบต้นแบบเพื่อการศึกษา — ทำนายจากสถิติก่อนการแข่งขัน (ไม่ใช่ระบบพยากรณ์ที่แม่นยำสำหรับใช้งานจริง)")

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("ทีม A (ทีมที่ต้องการทำนาย)", TEAMS, index=TEAMS.index("Brazil") if "Brazil" in TEAMS else 0)
with col2:
    default_b = "Germany" if "Germany" in TEAMS and "Germany" != team_a else TEAMS[1]
    remaining = [t for t in TEAMS if t != team_a]
    team_b = st.selectbox("ทีม B (คู่แข่ง)", remaining, index=remaining.index(default_b) if default_b in remaining else 0)

predict_clicked = st.button("ทำนายผล", type="primary", use_container_width=True)

if predict_clicked:
    if team_a == team_b:
        st.error("กรุณาเลือกสองทีมที่แตกต่างกัน")
    else:
        X_input = build_features(team_a, team_b)
        proba = model.predict_proba(X_input)[0]
        proba_map = dict(zip(model.classes_, proba))
        proba_series = pd.Series({c: proba_map.get(c, 0.0) for c in CLASS_ORDER})

        pred_class = proba_series.idxmax()
        confidence = proba_series.max()

        st.subheader(f"ผลการทำนาย: {team_a} vs {team_b}")

        proba_pct = (proba_series * 100).round(1)
        chart_df = pd.DataFrame({
            "ผล (มุมมองทีม A)": [f"{team_a} ชนะ", "เสมอ", f"{team_a} แพ้"],
            "ความน่าจะเป็น (%)": [proba_pct["Win"], proba_pct["Draw"], proba_pct["Loss"]],
        }).set_index("ผล (มุมมองทีม A)")
        st.bar_chart(chart_df)

        label_th = {"Win": f"{team_a} ชนะ", "Draw": "เสมอ", "Loss": f"{team_a} แพ้"}

        if confidence < THRESHOLD:
            st.info(
                f"**ระบบไม่มั่นใจในการทำนายคู่นี้** "
                f"(ความน่าจะเป็นสูงสุด {confidence*100:.1f}% ต่ำกว่าเกณฑ์ {THRESHOLD*100:.0f}%) "
                f"หากอ้างอิงผลที่เป็นไปได้มากที่สุด ระบบเอนไปทาง **{label_th[pred_class]}** "
            )
        else:
            st.success(f"✅ ระบบทำนายว่า **{label_th[pred_class]}** (ความมั่นใจ {confidence*100:.1f}%)")

        with st.expander("ดูค่า feature ที่ใช้คำนวณ"):
            st.dataframe(X_input.T.rename(columns={0: "ค่า (ทีม A - ทีม B)"}))

st.divider()
with st.expander("เกี่ยวกับระบบ / ข้อจำกัด"):
    st.markdown(
        f"""
- โมเดล: **Logistic Regression**
- Feature ที่ใช้: ผลต่างจำนวนครั้งเข้าร่วม/แชมป์/เข้าชิง/รอบรองฯ, ผลต่างอันดับ FIFA, สถิติการพบกัน (เฉพาะทีมที่เจอกันในรอบแบ่งกลุ่ม 2018)
- เกณฑ์ความเชื่อมั่น (confidence threshold) ปัจจุบัน: **{THRESHOLD*100:.0f}%** — ถ้าความน่าจะเป็นสูงสุดต่ำกว่านี้ ระบบจะแจ้งว่า "ไม่มั่นใจ" 
- ข้อจำกัดสำคัญ: ข้อมูลฝึกมีเพียง 48 นัด, โมเดลยังทำนายผล **เสมอ (Draw)** ได้ไม่ดี, สถิติ head-to-head มีเฉพาะคู่ทีมที่พบกันในรอบแบ่งกลุ่มปี 2018 เท่านั้น (คู่อื่นจะถือว่าไม่มีประวัติ = 0)
        """
    )
