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

## How

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

We have a model to predict the outcome of a match.

### Monte Carlo Simulation

Now that we can predict a match, we can simulate an entire World Cup.

For the Group Stage, we predict the 3 matches that each team plays and pick the top two teams of each group and the 8 best third-place finishers.
For the Knockout Stage, we predict the 16 matches in the Round of 32, the 8 matches in the Round of 16, the 4 matches in the quarter-finals, the 2 matches in the semi-finals, and finally the final.

From one simulation, we might predict Argentina to have won the final, but based on the outcome of the 2026 World Cup, that was not the case.
Therefore, we simulate the World Cup more than once. In my case, I simulated 20,000 World Cups. We track for each country whether they advance, reach the Round of 32, the Round of 16, the quarter-finals, the semi-finals, or win the title.
In the end, I was left with `predictions_2026.csv`.

### Implementation

I asked Claude to not only extract the features, but also write the entire source code.

## Result

See `predictions_2026.csv`.

Analyzing my predictions, it accurately predicted Spain to win the title and Argentina to come second.
Well, not directly, but it predicted that Spain was the most likely to win the title.

However, there are some predictions that were off the mark, like Brazil.
