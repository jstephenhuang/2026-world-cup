# World Cup 2026

Back when March Madness was happening, I participated in my class's NCAA basketball tournament bracket.
I do play basketball, but I have zero knowledge of collegiate basketball teams and players.
I did not want to create my bracket based on feeling, with zero strategy.
Therefore, I did a little research on some ways I could use machine learning to help me create my picks.
I stumbled upon random forests.
It seemed simple enough, and so I dove a little deeper into it, understanding the foundations of random forests.
Furthermore, there were multiple Kaggle datasets containing relevant data from previous NCAA tournaments.
Unfortunately, I did not have time to implement a working random forest to help with the bracket.
I ended up making my picks from pure feeling and instinct.
I placed low on the leaderboard.

That was March 2026.

With the 2026 FIFA World Cup, I had another chance at implementing my random forest.
Plus, coding agents have gotten so good that I could even ask them to refine and strengthen my predictions.

This repository attempts to predict how likely each country in the 2026 World Cup is to advance, reach the R16, reach the quarter-finals, reach the semi-finals, reach the final, and win the title.

## How it works

### What is a Random Forest (in my own words)

If I were to describe it in one sentence:

A Random Forest aggregates the predictions of x Decision Trees, all a little different from each other, resulting in a probability for each possible outcome.

More thorough explanation:

*A Decision Tree is a tree-like structure that predicts an outcome or answers a question by answering yes-or-no questions.*

In a Random Forest, each decision tree is trained on a random sample of the dataset it uses.
Therefore, some trees might put more importance on certain features than other trees.
For example, Tree 1 might strictly have nodes comparing the country's Elo, but Tree 2 might instead look at the country's average goals.
In this case, given Spain vs Brazil, Tree 1 might predict a win for Spain, but Tree 2 might predict a draw.
The Random Forest will essentially spawn x Decision Trees and stack the prediction from each tree.
In the end, it will output a chart-like result where, for each outcome, there is a count of how many trees predicted it:

```py
# for 600 trees
{"win": 422, "draw": 78, "loss": 100}
```

which you can convert to probabilities by dividing each count by the number of trees:

```py
{"win": 0.70333333333, "draw": 0.13, "loss": 0.16666666666}
```

Look at that, our random forest is able to predict the outcome of a match.

### Monte Carlo Simulation

Now that we can predict a match, we can simulate an entire World Cup.

For the Group Stage, we predict the 3 matches that each team plays and pick the top two teams of each group and the 8 best third-place finishers.
For the Knockout Stage, we predict the 16 matches in the Round of 32, the 8 matches in the Round of 16, the 4 matches in the quarter-finals, the 2 matches in the semi-finals, and finally the final.

From one simulation, we might predict Argentina to have won the final, but based on the outcome of the 2026 World Cup, that was not the case.
Therefore, we simulate the World Cup more than once. In my case, I simulated 20,000 World Cups. We track for each country whether they advance, reach the Round of 32, the Round of 16, the quarter-finals, the semi-finals, or win the title.
In the end, I was left with `predictions_2026.csv`.

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

Here was the results `predictions_2026.csv`:

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
Well, not directly, but it predicted that Spain was the most likely to win the title.

However, there are some predictions that were off the mark, like Brazil.

## Future improvements

For this run, I trained the random forest using only `results.csv`. If I had more time, I would train a larger forest (more decision trees) and incorporate the other files available in the Kaggle dataset.
More specifically, there were two other useful csvs, `goalscorers.csv` and `shootouts.csv`.
Including shootout data may have improved my predictions for matches decided by penalties.
For example, the random forest ranked Germany higher than Paraguay.
But Germany's penalty kicks against Paraguay were absolutely horrendous, which cost them the game.
But perhaps the pressure got to them, and that extra data could have wrongly predicted the ranking of other teams. :shrug:
