from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

data_path = Path("World_Cup_2018_Dataset.csv")
df = pd.read_csv(data_path)
df = df.dropna(subset=["Team"]).copy()


def normalize_team_name(x):
    fixes = {
        "Porugal": "Portugal",
        "IRAN": "Iran",
        "Costarica": "Costa Rica",
        "Columbia": "Colombia",
        "Korea": "South Korea",
    }
    x = str(x).strip()
    return fixes.get(x, x)


df["Team"] = df["Team"].map(normalize_team_name)

opponent_cols = [
    "First match \nagainst",
    "Second match\n against",
    "Third match\n against",
]
for col in opponent_cols:
    df[col] = df[col].map(normalize_team_name)

stat_cols = {
    "Previous \nappearances": "appearances",
    "Previous \ntitles": "titles",
    "Previous\n finals": "finals",
    "Previous\n semifinals": "semifinals",
    "Current \nFIFA rank": "fifa_rank",
}

team_stats = df.set_index("Team")[list(stat_cols.keys())].rename(columns=stat_cols)

h2h_specs = [
    ("First match \nagainst", "history with \nfirst opponent\n W-L", "history with\n first opponent\n goals"),
    ("Second match\n against", "history with\n second opponent\n W-L", "history with\n second opponent\n goals"),
    ("Third match\n against", "history with\n third opponent\n W-L", "history with\n third opponent\n goals"),
]

h2h_rows = []
for _, row in df.iterrows():
    for opp_col, wl_col, goals_col in h2h_specs:
        opponent = row[opp_col]
        if pd.isna(opponent):
            continue
        wl = 0.0 if pd.isna(row[wl_col]) else float(row[wl_col])
        goals = 0.0 if pd.isna(row[goals_col]) else float(row[goals_col])
        h2h_rows.append({"team": row["Team"], "opponent": opponent, "wl": wl, "goals": goals})

h2h_table = pd.DataFrame(h2h_rows).set_index(["team", "opponent"])


def get_h2h(team, opponent):
    try:
        rec = h2h_table.loc[(team, opponent)]
        return float(rec["wl"]), float(rec["goals"])
    except KeyError:
        return 0.0, 0.0


matches = [
    ("Russia", "Saudi Arabia", 5, 0), ("Egypt", "Uruguay", 0, 1),
    ("Russia", "Egypt", 3, 1), ("Uruguay", "Saudi Arabia", 1, 0),
    ("Uruguay", "Russia", 3, 0), ("Saudi Arabia", "Egypt", 2, 1),
    ("Morocco", "Iran", 0, 1), ("Portugal", "Spain", 3, 3),
    ("Portugal", "Morocco", 1, 0), ("Iran", "Spain", 0, 1),
    ("Iran", "Portugal", 1, 1), ("Spain", "Morocco", 2, 2),
    ("France", "Australia", 2, 1), ("Peru", "Denmark", 0, 1),
    ("Denmark", "Australia", 1, 1), ("France", "Peru", 1, 0),
    ("Denmark", "France", 0, 0), ("Australia", "Peru", 0, 2),
    ("Argentina", "Iceland", 1, 1), ("Croatia", "Nigeria", 2, 0),
    ("Argentina", "Croatia", 0, 3), ("Nigeria", "Iceland", 2, 0),
    ("Nigeria", "Argentina", 1, 2), ("Iceland", "Croatia", 1, 2),
    ("Costa Rica", "Serbia", 0, 1), ("Brazil", "Switzerland", 1, 1),
    ("Brazil", "Costa Rica", 2, 0), ("Serbia", "Switzerland", 1, 2),
    ("Serbia", "Brazil", 0, 2), ("Switzerland", "Costa Rica", 2, 2),
    ("Germany", "Mexico", 0, 1), ("Sweden", "South Korea", 1, 0),
    ("South Korea", "Mexico", 1, 2), ("Germany", "Sweden", 2, 1),
    ("South Korea", "Germany", 2, 0), ("Mexico", "Sweden", 0, 3),
    ("Belgium", "Panama", 3, 0), ("Tunisia", "England", 1, 2),
    ("Belgium", "Tunisia", 5, 2), ("England", "Panama", 6, 1),
    ("England", "Belgium", 0, 1), ("Panama", "Tunisia", 1, 2),
    ("Colombia", "Japan", 1, 2), ("Poland", "Senegal", 1, 2),
    ("Japan", "Senegal", 2, 2), ("Poland", "Colombia", 0, 3),
    ("Japan", "Poland", 0, 1), ("Senegal", "Colombia", 0, 1),
]

rows = []
for team, opponent, team_goals, opponent_goals in matches:
    team_row = team_stats.loc[team]
    opponent_row = team_stats.loc[opponent]
    team_wl, team_h2h_goals = get_h2h(team, opponent)
    opponent_wl, opponent_h2h_goals = get_h2h(opponent, team)

    if team_goals > opponent_goals:
        result = "Win"
    elif team_goals < opponent_goals:
        result = "Loss"
    else:
        result = "Draw"

    rows.append({
        "team": team,
        "opponent": opponent,
        "appearances_diff": team_row["appearances"] - opponent_row["appearances"],
        "titles_diff": team_row["titles"] - opponent_row["titles"],
        "finals_diff": team_row["finals"] - opponent_row["finals"],
        "semifinals_diff": team_row["semifinals"] - opponent_row["semifinals"],
        "fifa_rank_diff": team_row["fifa_rank"] - opponent_row["fifa_rank"],
        "h2h_wl_diff": team_wl - opponent_wl,
        "h2h_goals_diff": team_h2h_goals - opponent_h2h_goals,
        "match_result": result,
    })

model_df = pd.DataFrame(rows)

feature_cols = [
    "appearances_diff", "titles_diff", "finals_diff",
    "semifinals_diff", "fifa_rank_diff",
    "h2h_wl_diff", "h2h_goals_diff",
]

X = model_df[feature_cols]
y = model_df["match_result"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__class_weight": [None, "balanced"],
}
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
final_model = grid.best_estimator_

print("Best params:", grid.best_params_)
for name, Xp, yp in [("Train", X_train, y_train), ("Validation", X_val, y_val), ("Test", X_test, y_test)]:
    pred = final_model.predict(Xp)
    print(f"{name}: accuracy={accuracy_score(yp, pred):.3f} "
          f"macro_f1={f1_score(yp, pred, average='macro', zero_division=0):.3f}")

joblib.dump(final_model, "model.joblib")
joblib.dump(team_stats, "team_stats.joblib")
joblib.dump(h2h_table, "h2h_table.joblib")
joblib.dump({
    "feature_cols": feature_cols,
    "teams": sorted(team_stats.index.tolist()),
    "class_order": ["Win", "Draw", "Loss"],
    "confidence_threshold": 0.45,   
}, "meta.joblib")

print("\nบันทึกไฟล์ model.joblib, team_stats.joblib, h2h_table.joblib, meta.joblib")
