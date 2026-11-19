import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import poisson
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. DATA PREPARATION
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.dropna(
        subset=[
            "Date", "HomeTeam", "AwayTeam",
            "FTHG", "FTAG", "FTR"
        ]
    )
    .sort_values("Date")
    .reset_index(drop=True)
    .copy()
)

df["FTHG"] = df["FTHG"].astype(int)
df["FTAG"] = df["FTAG"].astype(int)


# ============================================================
# 2. HISTORICAL TEAM STRENGTHS
# ============================================================

def strengths_before(date, matches):

    past = matches[matches["Date"] < date]

    if len(past) == 0:
        return pd.DataFrame(
            columns=["Attack", "Defence"]
        ).rename_axis("Team")

    league_home_avg = past["FTHG"].mean()
    league_away_avg = past["FTAG"].mean()

    avg_team = (league_home_avg + league_away_avg) / 2

    teams = sorted(
        set(past["HomeTeam"]) | set(past["AwayTeam"])
    )

    rows = []

    for team in teams:

        home = past[past["HomeTeam"] == team]
        away = past[past["AwayTeam"] == team]

        scored = home["FTHG"].sum() + away["FTAG"].sum()
        conceded = home["FTAG"].sum() + away["FTHG"].sum()

        played = len(home) + len(away)

        attack = (scored / played) / avg_team
        defence = (conceded / played) / avg_team

        rows.append({
            "Team": team,
            "Attack": attack,
            "Defence": defence
        })

    return pd.DataFrame(rows).set_index("Team")


# ============================================================
# 3. RECENT FORM
# ============================================================

def recent_form(team, date, matches, n=5, venue=None):

    past = matches[matches["Date"] < date]

    if venue == "home":

        games = past[past["HomeTeam"] == team].tail(n)

        scored = games["FTHG"].to_numpy()
        conceded = games["FTAG"].to_numpy()

    elif venue == "away":

        games = past[past["AwayTeam"] == team].tail(n)

        scored = games["FTAG"].to_numpy()
        conceded = games["FTHG"].to_numpy()

    else:

        games = past[
            (past["HomeTeam"] == team) |
            (past["AwayTeam"] == team)
        ].tail(n)

        is_home = (games["HomeTeam"] == team).to_numpy()

        scored = np.where(
            is_home,
            games["FTHG"],
            games["FTAG"]
        )

        conceded = np.where(
            is_home,
            games["FTAG"],
            games["FTHG"]
        )

    if len(games) == 0:
        return {
            "games": 0,
            "goals_scored": np.nan,
            "goals_conceded": np.nan,
            "points": np.nan
        }

    points = np.where(
        scored > conceded,
        3,
        np.where(scored == conceded, 1, 0)
    )

    return {
        "games": len(games),
        "goals_scored": scored.mean(),
        "goals_conceded": conceded.mean(),
        "points": points.mean()
    }


# ============================================================
# 4. POISSON MATCH PREDICTION
# ============================================================

def predict_match(
    home_team,
    away_team,
    date,
    matches,
    form_weight=0.2,
    n_form=5,
    venue_form=False,
    max_goals=10
):

    past = matches[matches["Date"] < date]

    if len(past) == 0:
        raise ValueError("No historical matches available.")

    league_home_avg = past["FTHG"].mean()
    league_away_avg = past["FTAG"].mean()

    avg_team = (league_home_avg + league_away_avg) / 2

    strengths = strengths_before(date, matches)

    def blended(team, venue):

        if team not in strengths.index:
            attack, defence = 1.0, 1.0
        else:
            attack = strengths.loc[team, "Attack"]
            defence = strengths.loc[team, "Defence"]

        form = recent_form(
            team,
            date,
            matches,
            n_form,
            venue if venue_form else None
        )

        if form["games"] < n_form:
            return attack, defence

        recent_attack = form["goals_scored"] / avg_team
        recent_defence = form["goals_conceded"] / avg_team

        attack = (
            (1 - form_weight) * attack
            + form_weight * recent_attack
        )

        defence = (
            (1 - form_weight) * defence
            + form_weight * recent_defence
        )

        return attack, defence

    attack_h, defence_h = blended(home_team, "home")
    attack_a, defence_a = blended(away_team, "away")

    expected_home = (
        league_home_avg * attack_h * defence_a
    )

    expected_away = (
        league_away_avg * attack_a * defence_h
    )

    expected_home = max(expected_home, 1e-10)
    expected_away = max(expected_away, 1e-10)

    goals = np.arange(max_goals + 1)

    home_probs = poisson.pmf(goals, expected_home)
    away_probs = poisson.pmf(goals, expected_away)

    score_matrix = np.outer(home_probs, away_probs)

    score_matrix /= score_matrix.sum()

    home_win = np.tril(score_matrix, -1).sum()
    draw = np.trace(score_matrix)
    away_win = np.triu(score_matrix, 1).sum()

    return (
        expected_home,
        expected_away,
        home_win,
        draw,
        away_win
    )


# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

def generate_predictions(
    matches,
    prediction_matches,
    form_weight=0.2,
    n_form=5,
    venue_form=False
):

    predictions = []
    skipped = 0

    for _, match in prediction_matches.iterrows():

        try:

            expected_home, expected_away, pH, pD, pA = predict_match(
                match["HomeTeam"],
                match["AwayTeam"],
                match["Date"],
                matches,
                form_weight=form_weight,
                n_form=n_form,
                venue_form=venue_form
            )

            predictions.append({
                "Date": match["Date"],
                "HomeTeam": match["HomeTeam"],
                "AwayTeam": match["AwayTeam"],
                "FTR": match["FTR"],
                "FTHG": match["FTHG"],
                "FTAG": match["FTAG"],
                "ExpectedHomeGoals": expected_home,
                "ExpectedAwayGoals": expected_away,
                "pH": pH,
                "pD": pD,
                "pA": pA
            })

        except (KeyError, ValueError):
            skipped += 1

    predictions = pd.DataFrame(predictions)

    print("Matches available:", len(prediction_matches))
    print("Predictions generated:", len(predictions))
    print("Matches skipped:", skipped)

    return predictions


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate_model(predictions):

    actual = predictions["FTR"].to_numpy()

    probabilities = predictions[
        ["pH", "pD", "pA"]
    ].to_numpy()

    predicted = np.array(["H", "D", "A"])[
        np.argmax(probabilities, axis=1)
    ]

    loss = log_loss(
        actual,
        probabilities,
        labels=["H", "D", "A"]
    )

    accuracy = accuracy_score(actual, predicted)

    print("Log loss:", loss)
    print("Accuracy:", accuracy)

    print("\nClassification report:")

    print(
        classification_report(
            actual,
            predicted,
            labels=["H", "D", "A"],
            zero_division=0
        )
    )

    cm = confusion_matrix(
        actual,
        predicted,
        labels=["H", "D", "A"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Home", "Draw", "Away"]
    )

    display.plot()
    plt.show()

    return {
        "Log Loss": loss,
        "Accuracy": accuracy
    }


# ============================================================
# 7. HISTORICAL-FREQUENCY BASELINE
# ============================================================

def evaluate_baseline(matches, predictions):

    baseline_probs = []

    for date in predictions["Date"]:

        past = matches[matches["Date"] < date]

        frequencies = (
            past["FTR"]
            .value_counts(normalize=True)
            .reindex(["H", "D", "A"], fill_value=0)
        )

        baseline_probs.append(frequencies.to_numpy())

    baseline_probs = np.array(baseline_probs)

    baseline_probs = np.clip(
        baseline_probs,
        1e-10,
        1
    )

    baseline_probs /= baseline_probs.sum(axis=1, keepdims=True)

    loss = log_loss(
        predictions["FTR"],
        baseline_probs,
        labels=["H", "D", "A"]
    )

    print("Baseline log loss:", loss)

    return loss


# ============================================================
# 8. CALIBRATION ANALYSIS
# ============================================================

def calibration_analysis(predictions):

    calibration = predictions.copy()

    calibration["ProbabilityBin"] = pd.cut(
        calibration["pH"],
        bins=np.arange(0, 1.1, 0.1),
        include_lowest=True
    )

    results = calibration.groupby(
        "ProbabilityBin",
        observed=False
    ).agg(
        MeanPredicted=("pH", "mean"),
        ActualHomeWinRate=(
            "FTR",
            lambda x: (x == "H").mean()
        ),
        Matches=("FTR", "count")
    )

    print(results)

    plt.figure(figsize=(7, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        results["MeanPredicted"],
        results["ActualHomeWinRate"],
        marker="o",
        label="Model"
    )

    plt.xlabel("Mean predicted home-win probability")
    plt.ylabel("Actual home-win frequency")
    plt.title("Home-win calibration")

    plt.legend()
    plt.grid(True)
    plt.show()

    return results


# ============================================================
# 9. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_matches = df[
    df["Date"] < "2023-08-01"
]

val_matches = df[
    (df["Date"] >= "2023-08-01") &
    (df["Date"] < "2024-08-01")
]

test_matches = df[
    (df["Date"] >= "2024-08-01") &
    (df["Date"] < "2025-08-01")
]

print("Training matches:", len(train_matches))
print("Validation matches:", len(val_matches))
print("Test matches:", len(test_matches))


# ============================================================
# 10. GENERATE VALIDATION PREDICTIONS
# ============================================================

val_predictions = generate_predictions(
    df,
    val_matches,
    form_weight=0.2,
    n_form=5,
    venue_form=False
)

print("\nVALIDATION PERFORMANCE")

val_metrics = evaluate_model(val_predictions)


# ============================================================
# 11. GENERATE UNSEEN TEST PREDICTIONS
# ============================================================

test_predictions = generate_predictions(
    df,
    test_matches,
    form_weight=0.2,
    n_form=5,
    venue_form=False
)

print("\nTEST PERFORMANCE")

test_metrics = evaluate_model(test_predictions)


# ============================================================
# 12. BASELINE COMPARISON
# ============================================================

baseline_loss = evaluate_baseline(
    df,
    test_predictions
)

model_loss = test_metrics["Log Loss"]

print("\nMODEL COMPARISON")

print("Baseline log loss:", baseline_loss)
print("Model log loss:", model_loss)

improvement = (
    (baseline_loss - model_loss) / baseline_loss
) * 100

print("Relative log loss improvement:", improvement, "%")


# ============================================================
# 13. TEST CALIBRATION
# ============================================================

print("\nTEST CALIBRATION")

test_calibration = calibration_analysis(
    test_predictions
)


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

test_predictions.to_csv(
    "football_test_predictions.csv",
    index=False
)

print("\nPredictions saved successfully.")
