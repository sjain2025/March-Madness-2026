# March Madness Machine Learning (MMML) Model

## Summary

This project builds win-probability predictions for NCAA Division I tournament matchups using recency-weighted regular-season features, a **HistGradientBoostingClassifier**, and **IsotonicRegression** calibration. It writes prediction CSVs for every possible matchup, then uses **madness.py** to fill and print the 2026 men’s and women’s 68-team brackets.

=== Men's 2026 Regular Bracket ===

___Duke___                                                                                                                          _Arizona__
          |___Duke___                                                                                                    _Arizona__|          
__Siena___|          |                                                                                                  |          |LIU Brookl
                     |___Duke___                                                                              _Arizona__|                     
_Ohio St__           |          |                                                                            |          |           Villanova_
          |___TCU____|          |                                                                            |          |Villanova_|          
___TCU____|                     |                                                                            |                     |_Utah St__
                                |___Duke___                                                        _Arkansas_|                                
St John's_                      |          |                                                      |          |                      Wisconsin_
          |St John's_           |          |                                                      |          |           Wisconsin_|          
Northern I|          |          |          |                                                      |          |          |          |High Point
                     |St John's_|          |                                                      |          |_Arkansas_|                     
__Kansas__           |                     |                                                      |                     |           _Arkansas_
          |__Kansas__|                     |                                                      |                     |_Arkansas_|          
Cal Baptis|                                |                                                      |                                |__Hawaii__
                                           |Michigan S                                  _Arkansas_|                                           
Louisville                                 |          |                                |          |                                 ___BYU____
          |South Flor                      |          |                                |          |                      __Texas___|          
South Flor|          |                     |          |                                |          |                     |          |__Texas___
                     |Michigan S           |          |                                |          |           _Gonzaga__|                     
Michigan S           |          |          |          |                                |          |          |          |           _Gonzaga__
          |Michigan S|          |          |          |                                |          |          |          |_Gonzaga__|          
N Dakota S|                     |          |          |                                |          |          |                     |_Kennesaw_
                                |Michigan S|          |                                |          |__Purdue__|                                
___UCLA___                      |                     |                                |                     |                      _Miami FL_
          |___UCLA___           |                     |           _Florida__           |                     |           _Missouri_|          
___UCF____|          |          |                     |          |          |          |                     |          |          |_Missouri_
                     |Connecticu|                     |          |          |          |                     |__Purdue__|                     
Connecticu           |                                |          |          |          |                                |           __Purdue__
          |Connecticu|                                |          |          |          |                                |__Purdue__|          
__Furman__|                                           |          |          |          |                                           |Queens NC_
                                                      |_Florida__|          |_Arkansas_|                                                      
_Florida__                                            |                                |                                            _Michigan_
          |_Florida__                                 |                                |                                 _Michigan_|          
__Lehigh__|          |                                |                                |                                |          |___UMBC___
                     |_Florida__                      |                                |                      _Michigan_|                     
_Clemson__           |          |                     |                                |                     |          |           _Georgia__
          |___Iowa___|          |                     |                                |                     |          |_St Louis_|          
___Iowa___|                     |                     |                                |                     |                     |_St Louis_
                                |_Florida__           |                                |           _Michigan_|                                
Vanderbilt                      |          |          |                                |          |          |                      Texas Tech
          |Vanderbilt           |          |          |                                |          |          |           Texas Tech|          
McNeese St|          |          |          |          |                                |          |          |          |          |__Akron___
                     |Vanderbilt|          |          |                                |          |          |Texas Tech|                     
_Nebraska_           |                     |          |                                |          |                     |           _Alabama__
          |_Nebraska_|                     |          |                                |          |                     |_Alabama__|          
___Troy___|                                |          |                                |          |                                |_Hofstra__
                                           |_Florida__|                                |_Michigan_|                                           
North Caro                                 |                                                      |                                 Tennessee_
          |___VCU____                      |                                                      |                      Tennessee_|          
___VCU____|          |                     |                                                      |                     |          |___SMU____
                     |_Illinois_           |                                                      |           Tennessee_|                     
_Illinois_           |          |          |                                                      |          |          |           _Virginia_
          |_Illinois_|          |          |                                                      |          |          |_Virginia_|          
___Penn___|                     |          |                                                      |          |                     |Wright St_
                                |_Illinois_|                                                      |_Kentucky_|                                
St Mary's                       |                                                                            |                      _Kentucky_
          |St Mary's            |                                                                            |           _Kentucky_|          
Texas A&M_|          |          |                                                                            |          |          |Santa Clar
                     |St Mary's |                                                                            |_Kentucky_|                     
_Houston__           |                                                                                                  |           _Iowa St__
          |_Houston__|                                                                                                  |_Iowa St__|          
__Idaho___|                                                                                                                        |Tennessee 
                                                  Play-in game Y11: NC State vs Texas
                                                  Play-in game Z11: SMU vs Miami OH
                                                  Play-in game X16: Lehigh vs Prairie View
                                                  Play-in game Z16: Howard vs UMBC

Men's tiebreaker (predicted total points in championship): 140


=== Women's 2026 Regular Bracket ===

Connecticu                                                                                                                          ___UCLA___
          |Connecticu                                                                                                    ___UCLA___|          
UT San Ant|          |                                                                                                  |          |Cal Baptis
                     |Connecticu                                                                              ___UCLA___|                     
_Iowa St__           |          |                                                                            |          |           Oklahoma S
          |_Syracuse_|          |                                                                            |          |Oklahoma S|          
_Syracuse_|                     |                                                                            |                     |Princeton_
                                |Connecticu                                                        ___UCLA___|                                
_Maryland_                      |          |                                                      |          |                      Mississipp
          |_Maryland_           |          |                                                      |          |           Mississipp|          
Murray St_|          |          |          |                                                      |          |          |          |_Gonzaga__
                     |North Caro|          |                                                      |          |Mississipp|                     
North Caro           |                     |                                                      |                     |           Minnesota_
          |North Caro|                     |                                                      |                     |Minnesota_|          
W Illinois|                                |                                                      |                                |WI Green B
                                           |Connecticu                                  ___UCLA___|                                           
Notre Dame                                 |          |                                |          |                                 __Baylor__
          |Notre Dame                      |          |                                |          |                      __Baylor__|          
Fairfield_|          |                     |          |                                |          |                     |          |_Richmond_
                     |_Ohio St__           |          |                                |          |           ___Duke___|                     
_Ohio St__           |          |          |          |                                |          |          |          |           ___Duke___
          |_Ohio St__|          |          |          |                                |          |          |          |___Duke___|          
__Howard__|                     |          |          |                                |          |          |                     |Col Charle
                                |_Ohio St__|          |                                |          |___LSU____|                                
_Illinois_                      |                     |                                |                     |                      Texas Tech
          |_Colorado_           |                     |           Connecticu           |                     |           Texas Tech|          
_Colorado_|          |          |                     |          |          |          |                     |          |          |Villanova_
                     |Vanderbilt|                     |          |          |          |                     |___LSU____|                     
Vanderbilt           |                                |          |          |          |                                |           ___LSU____
          |Vanderbilt|                                |          |          |          |                                |___LSU____|          
High Point|                                           |          |          |          |                                           |Jacksonvil
                                                      |Connecticu|          |__Texas___|                                                      
South Caro                                            |                                |                                            __Texas___
          |South Caro                                 |                                |                                 __Texas___|          
Southern U|          |                                |                                |                                |          |Missouri S
                     |South Caro                      |                                |                      __Texas___|                     
_Clemson__           |          |                     |                                |                     |          |           __Oregon__
          |___USC____|          |                     |                                |                     |          |__Oregon__|          
___USC____|                     |                     |                                |                     |                     |Virginia T
                                |South Caro           |                                |           __Texas___|                                
Michigan S                      |          |          |                                |          |          |                      _Kentucky_
          |Michigan S           |          |          |                                |          |          |           _Kentucky_|          
Colorado S|          |          |          |          |                                |          |          |          |          |James Madi
                     |Michigan S|          |          |                                |          |          |_Kentucky_|                     
_Oklahoma_           |                     |          |                                |          |                     |           West Virgi
          |_Oklahoma_|                     |          |                                |          |                     |West Virgi|          
__Idaho___|                                |          |                                |          |                                |_Miami OH_
                                           |South Caro|                                |__Texas___|                                           
Washington                                 |                                                      |                                 _Alabama__
          |Washington                      |                                                      |                      _Alabama__|          
S Dakota S|          |                     |                                                      |                     |          |Rhode Isla
                     |Washington           |                                                      |           Louisville|                     
___TCU____           |          |          |                                                      |          |          |           Louisville
          |___TCU____|          |          |                                                      |          |          |Louisville|          
UC San Die|                     |          |                                                      |          |                     |_Vermont__
                                |___Iowa___|                                                      |Louisville|                                
_Georgia__                      |                                                                            |                      _NC State_
          |_Georgia__           |                                                                            |           Tennessee_|          
_Virginia_|          |          |                                                                            |          |          |Tennessee_
                     |___Iowa___|                                                                            |_Michigan_|                     
___Iowa___           |                                                                                                  |           _Michigan_
          |___Iowa___|                                                                                                  |_Michigan_|          
F Dickinso|                                                                                                                        |Holy Cross
                                                  Play-in game X16: Southern Univ vs Samford
                                                  Play-in game X10: Virginia vs Arizona St
                                                  Play-in game Y11: Nebraska vs Richmond
                                                  Play-in game Z16: Missouri St vs SF Austin

Women's tiebreaker (predicted total points in championship): 135

---

## Technologies Used

- **Language**: Python
- **Data & numerics**: **pandas**, **numpy**
- **ML & calibration**: **scikit-learn** — **HistGradientBoostingClassifier** (log-loss, tree-based), **IsotonicRegression** (probability calibration)
- **Validation**: walk-forward cross-validation over seasons to tune **recency-weighted** season importance
- **Data sources**: NCAA men’s and women’s regular-season and tournament CSVs in `data/` (compact and detailed results, seeds, and men’s rankings)

---

## Model Overview

The pipeline uses **gradient-boosted decision trees** (**HistGradientBoostingClassifier**) for **binary classification**: given a matchup, predict whether Team1 beats Team2. Training is **supervised** on historical tournament outcomes. Each game yields **two training examples** (winner vs loser and loser vs winner) so the model sees both orderings; the input vector is **differential features** (Team1 stats minus Team2 stats), which makes the problem symmetric and focuses the learner on relative strength. **Sample weighting** applies **exponential decay** by season (recent years weighted more) and by game order within a season (late-season games weighted more) to emphasize recency. **Walk-forward cross-validation** over seasons tunes the decay so that season weights minimize **log-loss** on held-out years. This is **tabular ML**—no natural language or computer vision; all inputs are numeric per-team and per-matchup summaries.

Raw classifier outputs are **probability-calibrated** with **IsotonicRegression** on a single held-out season so predicted win chances are better calibrated. The calibrated model scores all team pairs for the target season and writes prediction CSVs; **madness.py** then fills the 68-team men’s and women’s brackets from those probabilities.

---

## How to Run

Run everything from the `MMML` directory. Install dependencies: `pip install pandas numpy scikit-learn`.

1. **Generate 2026 prediction CSVs** (writes `MNCAATourneyPredictions.csv` and `WNCAATourneyPredictions.csv` into `predictions/`):

   ```bash
   python3 make_predictions.py --season 2026 --train_end_season 2025 --out_dir predictions
   ```

2. **Fill and print the 2026 brackets** (loads predictions, or generates them if missing; prints men’s and women’s brackets to the console):

   ```bash
   python3 madness.py
   ```
