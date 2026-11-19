# Football Match Outcome Prediction

A Python-based football match prediction project that estimates the
probabilities of a **home win, draw, and away win** using a Poisson goal
model. The project combines long-run team strength with recent form,
evaluates predictions using chronological backtesting, and includes a
separate workflow for tracking predictions on future matches.

The aim is not simply to predict the most likely result. The model
produces a probability distribution over the three outcomes, allowing
its performance to be assessed with proper probabilistic metrics such as
log loss.

------------------------------------------------------------------------

## Contents

-   [Project overview](#project-overview)
-   [Model methodology](#model-methodology)
-   [Features](#features)
-   [Evaluation methodology](#evaluation-methodology)
-   [Results](#results)
-   [Prospective live testing](#prospective-live-testing)
-   [How to run](#how-to-run)
-   [Project structure](#project-structure)
-   [Limitations and future
    improvements](#limitations-and-future-improvements)

------------------------------------------------------------------------

## Project overview

For each fixture, the model estimates the expected number of goals
scored by the home and away teams. It then uses independent Poisson
distributions to calculate the probability of each possible scoreline
and aggregates these into:

-   **Home win (H)**
-   **Draw (D)**
-   **Away win (A)**

The model uses information available before the fixture date only. This
is important: a historical prediction must not use the result of the
match being predicted or any later match.

### Main objectives

1.  Build a probabilistic football prediction model from first
    principles.
2.  Incorporate both overall team strength and recent performance.
3.  Tune model parameters using a validation season.
4.  assess performance on a later, unseen season.
5.  Track predictions made before future matches to measure prospective
    performance.

------------------------------------------------------------------------

## Model methodology

### 1. Overall team strength

The model estimates each team's attacking and defensive strength using
matches played before the fixture date.

-   **Attack strength** represents a team's goalscoring ability relative
    to the league.
-   **Defensive strength** represents how many goals a team tends to
    concede relative to the league.

The estimates are recalculated using only historical matches available
before each prediction.

### 2. Recent form

Recent form is calculated over the team's previous five matches. It
includes:

-   Average goals scored.
-   Average goals conceded.
-   Points earned.

The model blends the overall team-strength estimates with recent-form
information. This allows the prediction to respond to more recent
performances without relying exclusively on a small number of matches.

### 3. Expected goals and Poisson distributions

The model calculates expected goals, usually written as
(`\lambda`{=tex}\_H) for the home team and (`\lambda`{=tex}\_A) for the
away team.

The number of goals scored by each team is modelled using a Poisson
distribution:

\[ P(X=k)=`\frac{e^{-\lambda}\lambda^k}{k!}`{=tex} \]

The model assumes that home and away goal counts are conditionally
independent:

\[ P(H=i,A=j)=P(H=i)P(A=j) \]

A scoreline probability matrix is constructed over a range of possible
goal counts. The probabilities are then summed according to the match
outcome:

-   Home win: (i\>j)
-   Draw: (i=j)
-   Away win: (i\<j)

The resulting three probabilities are normalised to sum to one.

### 4. Selected model configuration

The best configuration found during validation was:

  Parameter                           Selected value
  ---------------------------- ---------------------
  Recent-form weight                             0.2
  Recent-form window                       5 matches
  Venue-specific recent form                Disabled
  Goal model                     Independent Poisson

Venue-specific recent form was tested but did not improve validation log
loss, so it was excluded from the selected configuration.

------------------------------------------------------------------------

## Features

-   Historical team attack and defence estimates.
-   Recent-form features based on the previous five matches.
-   Optional venue-specific form, evaluated during model selection.
-   Poisson scoreline probability matrix.
-   Aggregated home/draw/away probabilities.
-   Historical-frequency baseline for comparison.
-   Log loss and accuracy evaluation.
-   Calibration analysis.
-   Chronological train/validation/test workflow.
-   Separate prospective prediction tracker for upcoming fixtures.

------------------------------------------------------------------------

## Evaluation methodology

Random train/test splitting is inappropriate for this time-dependent
problem because it can allow information from later matches to influence
predictions for earlier matches.

Instead, the project uses a chronological evaluation:

  -----------------------------------------------------------------------
  Dataset                 Period                  Purpose
  ----------------------- ----------------------- -----------------------
  Training history        Before 1 August 2023    Estimate team strengths
                                                  and build historical
                                                  information

  Validation              1 August 2023 -- 31     Compare configurations
                          July 2024               and select model
                                                  parameters

  Test                    1 August 2024 -- 31     Evaluate the selected
                          July 2025               model on unseen
                                                  fixtures
  -----------------------------------------------------------------------

For each fixture, team statistics are calculated using matches dated
strictly before that fixture. Matches on the same date are excluded,
which is a conservative way to avoid accidentally using information that
may not have been available before kick-off.

### Metrics

**Log loss** is the main evaluation metric. It penalises confident
incorrect predictions more heavily than uncertain predictions. Lower is
better.

For a match with actual outcome (y), and predicted probability (p_y)
assigned to that outcome:

\[ L=-`\log`{=tex}(p_y) \]

The average log loss is calculated over all evaluated matches.

**Accuracy** is the proportion of fixtures for which the outcome with
the highest predicted probability matches the actual result. It is
useful as a secondary metric, but it does not assess the quality of the
full probability distribution.

**Calibration** measures whether predicted probabilities correspond to
observed frequencies. For example, among matches assigned a 60% home-win
probability, approximately 60% should result in a home win over a
sufficiently large sample.

### Baseline

The project compares the Poisson model with a simple
historical-frequency baseline. For each fixture, the baseline estimates
home/draw/away probabilities from the outcomes of matches played before
that fixture.

A second comparison is made against the Poisson model without the
recent-form adjustment. This helps isolate whether adding recent form
improves the model.

------------------------------------------------------------------------

## Results

### Validation results

The following figures are from the 2023/24 validation period. They were
used for model selection and should not be presented as final
unseen-test performance.

  -----------------------------------------------------------------------
  Model / configuration                               Validation log loss
  ------------------------------ ----------------------------------------
  Historical outcome-frequency                                   1.055002
  baseline                       

  Poisson model without                                          0.937189
  recent-form adjustment         

  Poisson model, recent-form                                     0.928731
  weight 0.3, 5-match window     

  **Selected Poisson model,                                  **0.927123**
  weight 0.2, 5-match window**   
  -----------------------------------------------------------------------

The selected model improved validation log loss by approximately
**12.1%** relative to the historical-frequency baseline.

The selected configuration also performed better than the tested
alternatives. Venue-specific recent form was not retained: its
validation log loss was 0.934778, compared with 0.927123 without
venue-specific form.

### Unseen test results (2024/25)

> **To complete after running the final test evaluation.** Do not tune
> the model using this test period; it is intended to provide an
> unbiased assessment of the selected configuration.

  Metric                     Selected model   Historical-frequency baseline
  ------------------------ ---------------- -------------------------------
  Number of test matches                ---                             ---
  Log loss                              ---                             ---
  Accuracy                              ---                             ---

**Test-set improvement in log loss:** to be calculated after evaluation.

### Calibration

Calibration should be assessed on the final selected configuration and
reported separately for the validation and test periods where possible.

A calibration plot should compare the mean predicted probability in each
bin with the observed outcome frequency. Bins containing few matches
should be interpreted cautiously.

### Interpretation

The validation results suggest that:

-   The Poisson model provides more informative probabilities than the
    simple historical-frequency baseline.
-   Incorporating recent form improves validation performance compared
    with the version without the recent-form adjustment.
-   The five-match recent-form window and 0.2 blend weight gave the best
    result among the configurations tested.
-   Venue-specific recent form did not improve validation performance in
    the experiments conducted.

These are validation findings, not proof that the model will perform
equally well on future fixtures. The untouched test period and
prospective live results are needed to assess generalisation.

------------------------------------------------------------------------

## Prospective live testing

In addition to historical backtesting, the project includes a separate
notebook for recording predictions on upcoming fixtures.

The workflow is:

1.  Load the historical match data.
2.  Enter upcoming fixtures before kick-off.
3.  Generate home/draw/away probabilities using the fixed selected
    configuration.
4.  Save each prediction, its timestamp, and the baseline probabilities
    to a CSV file.
5.  After the match, enter the actual result.
6.  Evaluate completed predictions using log loss, accuracy, and
    calibration.

The live tracker stores predictions so that they can be evaluated later.
Existing predictions should not be overwritten when results become
available.

**Important testing rules:**

-   Generate and save predictions before kick-off.
-   Do not change the model parameters after seeing the outcomes of
    tracked matches.
-   Do not retrospectively replace a saved prediction with a newer one.
-   Keep the original probabilities even after updating the historical
    match dataset.
-   Record enough fixtures to obtain a meaningful sample; a small number
    of matches can produce highly variable results.

Prospective testing is particularly useful because it tests the complete
prediction process on genuinely future matches, rather than only
reconstructing predictions from historical data.

------------------------------------------------------------------------

## How to run

### Requirements

The project uses Python and common data-science libraries:

-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib` (for visualisations, if used)
-   `jupyter` (to run the notebooks)

Install the dependencies with:

``` bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Historical evaluation

1.  Clone or download this repository.
2.  Place the historical match data in the project directory.
3.  Update the data path in the notebook or script.
4.  Run the model and evaluation cells in order.
5.  Use the validation period to select the configuration.
6.  Run the final test evaluation once, with the selected configuration
    fixed.

### Live testing

Open `football_live_testing.ipynb` and:

1.  Set `DATA_PATH` to the location of your historical match CSV.
2.  Load the historical data.
3.  Replace the example upcoming fixtures with real fixtures.
4.  Run the prediction cells before kick-off.
5.  After each fixture, enter the actual result and run the evaluation
    cells.

The live notebook writes predictions and completed results to
`live_predictions.csv`. Keep this file safe: it contains the original
probabilities needed to evaluate prospective performance.

------------------------------------------------------------------------

## Project structure

A suggested repository structure is:

``` text
football-match-prediction/
├── README.md
├── data/
│   └── matches.csv
├── notebooks/
│   ├── football_model.ipynb
│   └── football_live_testing.ipynb
├── live_predictions.csv
└── requirements.txt
```

The exact structure may differ depending on where the notebooks and data
are stored. Large raw datasets and generated files do not necessarily
need to be committed to GitHub.

------------------------------------------------------------------------

## Limitations and future improvements

The current model is deliberately kept relatively simple so that its
assumptions and evaluation are transparent.

Potential limitations include:

-   **Independent goal counts:** home and away goals may not be fully
    independent.
-   **Poisson assumption:** real football scores may not follow a
    Poisson distribution perfectly, particularly for low-scoring
    outcomes.
-   **Limited features:** the current model focuses on historical goals,
    team strength, and recent form. It does not yet incorporate
    injuries, line-ups, player quality, expected goals (xG), or market
    odds.
-   **Changing team strength:** simple historical averages may react
    slowly to major changes in a team.
-   **Sample size:** calibration and accuracy estimates can be noisy,
    especially for less common outcomes.

Possible future extensions include:

1.  Compare against a Dixon--Coles model to account for dependence in
    low-scoring results.
2.  Incorporate expected-goals data or other pre-match features, if
    reliable historical data is available.
3.  Compare against bookmaker-implied probabilities as an additional
    benchmark.
4.  Examine performance by season, league, and outcome class.
5.  Add confidence intervals or bootstrap uncertainty estimates.
6.  Improve the live tracker with automated fixture and result
    ingestion.

Any extension should be judged by whether it improves performance on
data that was not used to design or tune it.

------------------------------------------------------------------------

## Summary

This project builds a probabilistic football match prediction model
using team attack and defence estimates, recent form, and independent
Poisson goal distributions. It uses chronological validation to select
its configuration and reserves a later season for unseen testing. A
separate prospective tracker records predictions before future matches
so that real-world performance can be assessed without changing
historical predictions.
