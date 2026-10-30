# NBA Role Map

Groups NBA players into data-driven "positions" based on how they actually play, instead of the traditional PG/SG/SF/PF/C labels — then draws an interactive map of the results. This is a collapsed version of a different project currently in progress. 

Built entirely from public play-by-play data ([shufinskiy/nba_data](https://github.com/shufinskiy/nba_data)), no tracking data needed.

## How it works

1. **Builds a style profile for each player** (per 75 possessions): shot diet (rim, paint, midrange, corner 3, above-the-break 3), assisted vs. self-created shots, playmaking, turnovers, rebounding, blocks, steals, free-throw rate, putbacks, and transition.
2. **Adds a size index** based on who each player is matched up against defensively.
3. **Picks the number of positions automatically** using split-half stability: it clusters players using odd games and even games separately and chooses the setup that gives the most consistent groupings.
4. **Clusters players** with a Gaussian Mixture Model (GMM) and a Bayesian GMM, so each player gets a main position plus a membership score for every position.
5. **Names each position** from its strongest traits, e.g. *"Guard: playmaker / self-creator"* or *"Big: offensive glass / rim protector"*.

## Setup

```bash
pip install numpy pandas scikit-learn
```

## Usage

```bash
python nba_role_mapper.py --season 2023                 # 2023-24, both models
python nba_role_mapper.py --season 2023 --model bgmm --k 12
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--season` | 2023 | Season start year (2023 = 2023-24). Matchup data starts in 2017-18. |
| `--model` | both | `gmm`, `bgmm`, or `both` |
| `--k` | auto | Fix the number of positions |
| `--size-weight` | auto | Fix how much the size index counts |
| `--min-minutes` | 800 | Minimum minutes to include a player |
| `--data-dir` | data | Where downloaded data is cached |
| `--out-dir` | output | Where results are saved |

Data for the chosen season is downloaded automatically the first time (2023-24 is already included in `data/`).

## Output

Saved to `output/` for each model:

- `role_map_<season>_<model>.html` — interactive role map (open in a browser)
- `player_positions_<season>_<model>.csv` — each player's position, size index, and membership in every position
- `stability_<season>_<model>.csv` — stability scores for every setting tried

Results for 2023-24 are included.

## Limitations

No tracking data, so things like potential assists, time on the ball, and play types aren't captured.
