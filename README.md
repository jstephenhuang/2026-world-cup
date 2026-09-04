# World Cup 2026

Back when March Madness was happening, I participated in my class's NCAA basketball tournament bracket.
I do play basketball, but I have zero knowledge of collegiate basketball teams and players.
I did not want to create my bracket based on feeling, with zero strategy.
Therefore, I did a little research on some ways I could use machine learning to help me create my picks.
I stumbled upon random forests.
It seemed simple and intuitive, and so I dove a little deeper into it, understanding the foundations of random forests.
Furthermore, there were multiple Kaggle datasets containing relevant data from previous NCAA tournaments.
Unfortunately, I did not have time to implement a working random forest to help with the bracket.
I ended up making my picks from pure feeling and instinct.
I placed low on the leaderboard.

That was March 2026.

With the 2026 FIFA World Cup, I had another chance at implementing my random forest.
Plus, coding agents have gotten so good that I could even ask them to refine and strengthen my predictions.

This repository attempts to predict how likely each country in the 2026 World Cup is to advance, reach the R16, reach the quarter-finals, reach the semi-finals, reach the final, and win the title.

## How it works

### What is a Random Forest? (in my own words)

If I were to describe it in one sentence:

A **Random Forest** aggregates the predictions of `x` **Decision Trees**, all a little different from each other, resulting in a probability for each possible outcome.

More thorough explanation:

To understand a random forest, it is important to understand its fundamental component: a **Decision Tree**. A decision tree starts with a root node (a root question). Given the inputs, each answer takes the tree farther down until it reaches a leaf node (a possible output).

_A **Decision Tree** is a tree-like structure that predicts an outcome by asking a sequence of yes-or-no questions at its nodes._

For example:

<img width="800" height="400" alt="Example decision tree that predicts whether a customer will buy a product" src="https://github.com/user-attachments/assets/1940f5b6-9c51-482d-bc93-dcb79d8d65e8" />

---

So a random forest is essentially composed of multiple trees, making a **forest**.
Why random? Each tree is trained on a **random** sample / subset of the dataset. This makes the trees a little different from each other.

### How does a Random Forest work?

When designing a random forest, the inputs and outputs of the decision trees need to be defined in a way that lets them predict the outcome in question.

In my case, I want to predict the outcome of a soccer game (yes, I call it soccer, fight me) given the home team and away team.

The outputs are simple:

- home team win
- draw
- away team win

The inputs, on the other hand, require a little more work because they depend on the data.
Data is not usually processed in a form that a model can use directly, so it requires an extra step to turn it into something more usable.
I call this step **extracting features**.
For example, in the picture above, when trying to predict whether a customer will buy a product, age was one of the extracted feature from the customer data.
Why? Age, based on the data, was deemed relevant by the creator of the tree in predicting if a custom will buy a product.
So features could be human extracted, or nowadays by llms.

In this repo, I asked Claude to extract the features given the kaggle dataset, it extracted 9:

1. `elo_diff`: home team elo rating minus away team Elo rating
2. `home_elo`: home team elo rating before the match
3. `away_elo`: away team elo rating before the match
4. `neutral`: `1` if played at a neutral venue, otherwise `0`
5. `is_tournament`: `1` if the match was not a friendly, otherwise `0`
6. `home_form_gf`: home team's average goals scored over its previous five matches
7. `home_form_ga`: home team's average goals conceded over its previous five matches
8. `away_form_gf`: away team's average goals scored over its previous five matches
9. `away_form_ga`: away team's average goals conceded over its previous five matches

I will take some time to explain how these feature were extracted from the kaggle [International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) dataset. 

`elo_rating`:

To extract each country's elo rating from the kaggle dataset, the program starts every team at **1500 elo** and replays every historical match in chronological order.

Before each match, it saves both teams' ratings as `home_elo` and `away_elo`. Then, it calculates how likely the home team is to win:

```text
home_advantage = 65
expected_home_score = 1 / (1 + 10^((away_elo - (home_elo + home_advantage)) / 400))
```

This is the common logistic curve used on elo rating systems. 
For some intuition, if the `away_elo` is greater than `home_elo` + `home_advantage` (away team is stronger than home team), the exponent on 10 is positive which makes the denominator greater and the `expected_home_score` lower.
This makes sense since we would expect the away team to win (home team to lose) if they are way skilled and better than the home team even on home soil. 
This equation also keeps the the score between 0 and 1.

*The model uses a 65-point home advantage, unless the match is played on neutral ground in which case is 0.*

After the match, both ratings are updated based on how much the result differed from expectation:

```text
change =
30 × (1 + ln(1 + goal_difference))
× (actual_result - expected_result)
```

where `actual_result` is:

```text
home win = 1.0
draw     = 0.5
away win = 0.0
```

For example, if two 1500-rated teams play on neutral ground, the expected score is `0.5`. If the home team wins **2–0**:

```text
change = 30 × (1 + ln(3)) × (1 - 0.5)
       ≈ 31.48
```

So the winner becomes roughly **1531.5 elo**, and the loser becomes roughly **1468.5 elo**.

`home_form_gf`:

TODO...

---

So the winner becomes roughly **1531.5 elo**, and the loser becomes roughly **1468.5 elo**.

After each feature is well defined and extracted, we can use every historical match as a training example to build the decision trees.

To build one Decision Tree, the model starts at the root node and finds a yes-or-no question about a feature that best separates the different match outcomes.
It then repeats this process for each branch until it reaches a leaf node that predicts a home win, draw, or away win.

For example, one tree might first ask, "Is the home team's elo rating greater than 1800?"
It could then ask a different question depending on the answer, such as "Is the home team's average goals scored over the previous five matches greater than 1.5?"

Rather than trusting just one tree, the Random Forest builds `x` slightly different Decision Trees.
Each tree is trained on a random sample of the historical matches and considers a random subset of features at each node.
Therefore, some trees might put more importance on certain features than other trees.
For example, Tree 1 might strictly have nodes comparing the countries' elo ratings, but Tree 2 might instead look at their average goals.
Furthermore, some tree might have slightly different input feature values than other trees.
For example, Tree 1 might have a elo rating of 2111 for Spain, but Tree 2 have a elo rating of 2001. 

Before predicting a match, every tree receives the same nine features as inputs.
In a Spain vs. Brazil match, Tree 1 might predict a Spain win, but Tree 2 might predict a draw and maybe Tree 2 predicts Brazil to win.
The Random Forest aggregates all `x` tree's prediction.

As a result, we are left with:

```py
# for 600 trees
{"home win": 422, "draw": 78, "away win": 100}
```

which we can convert to probabilities by dividing each count by the number of trees:

```py
{"home win": 0.70333333333, "draw": 0.13, "away win": 0.16666666666}
```

Look at that, our random forest is able to predict the outcome of a match.

Lastly, the repo adds a calibration step that checks if this so its final probabilities are not literally just the raw tree-vote fractions. But this is the basic idea: the forest combines many slightly different opinions into probabilities for a match outcome.


### Monte Carlo Simulation

Now that we can predict a match, we can simulate an entire World Cup.

For the Group Stage, we predict the 3 matches that each team plays and pick the top two teams of each group and the 8 best third-place finishers.
For the Knockout Stage, we predict the 16 matches in the Round of 32, the 8 matches in the Round of 16, the 4 matches in the quarter-finals, the 2 matches in the semi-finals, and finally the final.

From one simulation, we might predict Argentina to have won the final, but based on the outcome of the 2026 World Cup, that was not the case.
Therefore, we simulate the World Cup more than once. In my case, I simulated 20,000 World Cups. We track, for each country, whether they advance, reach the Round of 16, the quarter-finals, the semi-finals, the final, or win the title.
In the end, I was left with [predictions_2026.csv](predictions_2026.csv).

### Implementation

I asked Claude to not only extract the features that the decision trees should use, but I also instructed it to write the entire source code.

## How to use

You need Python 3.14 or newer and [uv](https://docs.astral.sh/uv/) installed. The first command below will create the project's environment and install its dependencies automatically.

1. Download `results.csv` from Kaggle's [International football results from 1872 to 2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) dataset. The input CSV must include `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, and `neutral` columns.
2. From the repository root, run the simulation, passing the path to the downloaded CSV:

   ```bash
   uv run python main.py --data path/to/results.csv
   ```

   This trains the default Random Forest and runs 20,000 simulated tournaments. To choose a different number of simulations, use `--sims`:

   ```bash
   uv run python main.py --data path/to/results.csv --sims 50000
   ```

*More simulations do not necessarily result in better predictions. They reduce randomness in the estimated probabilities (essentially averaging your predictions), so after a certain point, additional simulations produce nearly the same results.*

3. When the run finishes, it writes the complete probabilities to `predictions_2026.csv` in the repository root. You can also use `--top 48` to print every team's result to the terminal.

## Result

Here are the results from [predictions_2026.csv](predictions_2026.csv):

| Team | Win title | Reach final | Reach semi | Reach quarter | Reach R16 | Advance group |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Spain | 15.67% | 24.22% | 35.90% | 52.11% | 73.79% | 97.43% |
| Argentina | 15.31% | 23.95% | 36.23% | 53.25% | 74.76% | 97.36% |
| France | 7.92% | 14.21% | 23.88% | 39.42% | 62.85% | 92.16% |
| Brazil | 6.43% | 12.53% | 23.70% | 40.64% | 66.31% | 96.09% |
| Portugal | 6.13% | 12.03% | 22.12% | 38.59% | 64.25% | 95.05% |
| Germany | 6.13% | 12.16% | 22.66% | 40.16% | 66.64% | 97.50% |
| Colombia | 3.98% | 8.48% | 16.73% | 31.98% | 56.22% | 90.65% |
| England | 3.73% | 8.16% | 16.45% | 30.96% | 56.62% | 91.68% |
| Belgium | 3.53% | 7.72% | 15.72% | 31.36% | 57.32% | 92.08% |
| Mexico | 3.41% | 8.11% | 17.80% | 36.92% | 67.11% | 96.21% |
| Morocco | 3.18% | 6.86% | 14.50% | 28.50% | 53.16% | 88.51% |
| Netherlands | 3.01% | 6.85% | 14.45% | 29.42% | 55.52% | 93.43% |
| Japan | 2.86% | 6.34% | 13.72% | 28.01% | 53.41% | 89.81% |
| Ecuador | 2.85% | 5.99% | 12.29% | 23.94% | 45.49% | 80.64% |
| Norway | 2.05% | 4.57% | 10.34% | 22.04% | 43.73% | 79.79% |
| Switzerland | 1.84% | 4.36% | 10.57% | 23.96% | 49.89% | 91.53% |
| Croatia | 1.66% | 3.89% | 9.34% | 20.87% | 43.50% | 84.36% |
| Denmark | 1.54% | 4.09% | 9.42% | 21.16% | 46.34% | 89.02% |
| Algeria | 1.43% | 3.51% | 8.57% | 19.90% | 44.95% | 87.11% |
| Austria | 1.25% | 3.30% | 8.81% | 20.35% | 45.21% | 90.86% |
| Uruguay | 1.19% | 3.20% | 8.04% | 19.09% | 43.62% | 88.43% |
| Canada | 0.92% | 2.88% | 7.14% | 17.75% | 41.18% | 87.02% |
| Senegal | 0.66% | 1.77% | 4.99% | 12.99% | 31.86% | 72.06% |
| Paraguay | 0.59% | 1.58% | 4.25% | 10.74% | 26.53% | 61.84% |
| Iran | 0.52% | 1.33% | 3.77% | 10.04% | 24.91% | 59.31% |
| Ivory Coast | 0.30% | 0.92% | 2.77% | 7.90% | 20.45% | 52.75% |
| Nigeria | 0.28% | 0.90% | 2.80% | 8.59% | 23.15% | 62.58% |
| South Korea | 0.28% | 0.98% | 3.08% | 8.49% | 22.01% | 55.92% |
| Egypt | 0.27% | 0.85% | 3.17% | 9.56% | 27.73% | 76.73% |
| Australia | 0.26% | 0.90% | 2.85% | 8.03% | 20.93% | 54.46% |
| United States | 0.19% | 0.76% | 2.68% | 8.24% | 23.59% | 61.54% |
| Uzbekistan | 0.17% | 0.47% | 1.69% | 5.38% | 16.49% | 50.88% |
| Sweden | 0.14% | 0.58% | 2.01% | 6.58% | 20.01% | 59.52% |
| Tunisia | 0.11% | 0.32% | 1.17% | 4.83% | 17.59% | 64.52% |
| Panama | 0.11% | 0.39% | 1.44% | 5.01% | 15.53% | 47.41% |
| South Africa | 0.04% | 0.10% | 0.53% | 2.23% | 9.37% | 42.49% |
| Wales | 0.04% | 0.22% | 1.03% | 3.97% | 14.12% | 51.77% |
| Cameroon | 0.04% | 0.11% | 0.51% | 2.23% | 8.61% | 33.67% |
| Jordan | 0.02% | 0.16% | 0.77% | 3.28% | 12.00% | 46.95% |
| Honduras | 0.01% | 0.10% | 0.42% | 1.80% | 7.66% | 34.50% |
| Haiti | 0.01% | 0.04% | 0.34% | 1.71% | 7.56% | 33.69% |
| Cape Verde | 0.01% | 0.03% | 0.16% | 0.96% | 4.27% | 21.73% |
| New Zealand | 0.01% | 0.01% | 0.10% | 0.84% | 4.04% | 20.13% |
| Qatar | 0.00% | 0.01% | 0.05% | 0.35% | 1.98% | 13.60% |
| Ghana | 0.00% | 0.02% | 0.17% | 0.96% | 5.46% | 30.14% |
| Saudi Arabia | 0.00% | 0.04% | 0.23% | 1.05% | 4.23% | 19.18% |
| Curacao | 0.00% | 0.00% | 0.04% | 0.22% | 1.42% | 11.86% |

Analyzing my predictions, it accurately predicted Spain to win the title and Argentina to come second.
Not explicitly, but it predicted that Spain was the most likely team to win the title.

There are some predictions that were off the mark, like Brazil.

I was able to use these results to win a small prediction pool of 25 people organized by the company (fellow.ai) I interned at over the summer.
It was an extremely fun experience seeing, in real time, how a set of yes-or-no questions was able to help me correctly predict most of the games and notably the title match.
I was the only one in the pool to have picked Spain to win the final.

I learned a lot and will definitely try using this model when March Madness 2027 comes around to see whether this was just a fluke or a reliable predictive model.

## Future improvements

For this run, I trained the random forest using only `results.csv`. If I had more time, I would train a larger forest (more decision trees) and incorporate the other files available in the Kaggle dataset.
More specifically, there were two other useful csvs, `goalscorers.csv` and `shootouts.csv`.
Including shootout data may have improved my predictions for matches decided by penalties.
For example, the random forest ranked Germany higher than Paraguay.
But Germany's penalty kicks against Paraguay were absolutely horrendous, which cost them the game.
But perhaps the pressure got to them, and that extra data could have wrongly predicted the ranking of other teams. 🤷
