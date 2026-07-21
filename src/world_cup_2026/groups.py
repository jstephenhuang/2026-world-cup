"""The 2026 World Cup groups (drawn Dec 5, 2025).

EDIT THIS to match the official draw / your bracket. Team names MUST match the
spelling used in the Kaggle results.csv (e.g. "United States", "South Korea",
"IR Iran" -> check the dataset; Kaggle uses "United States", "South Korea",
"Iran"). If a name doesn't match, that team falls back to a 1500 Elo default
and you'll get garbage for it — so verify the spellings.

This file ships with a PLACEHOLDER set of 12 groups using strong recent teams
so the script runs out of the box. Replace with the real draw.
"""

GROUPS_2026 = {
    "A": ["Mexico", "South Korea", "Norway", "Ivory Coast"],
    "B": ["Canada", "Switzerland", "Egypt", "Qatar"],
    "C": ["United States", "Japan", "Senegal", "Panama"],
    "D": ["Argentina", "Croatia", "Nigeria", "New Zealand"],
    "E": ["France", "Mexico", "Ecuador", "Saudi Arabia"],
    "F": ["Spain", "Uruguay", "Ghana", "Jordan"],
    "G": ["Brazil", "Denmark", "South Africa", "Honduras"],
    "H": ["England", "Colombia", "Cameroon", "Uzbekistan"],
    "I": ["Portugal", "Sweden", "Morocco", "Cape Verde"],
    "J": ["Germany", "Austria", "Tunisia", "Curacao"],
    "K": ["Netherlands", "Wales", "Algeria", "Haiti"],
    "L": ["Belgium", "Paraguay", "Australia", "Iran"],
}

# NOTE: a couple of teams are repeated in this placeholder for illustration.
# A real draw has 48 DISTINCT teams. Fix when you enter the actual groups.
