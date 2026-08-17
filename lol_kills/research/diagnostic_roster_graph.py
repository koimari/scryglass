"""Chronological roster graph diagnostic for the public Draft Score campaign."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lol_kills.research.atomized_rf_composite import LINEUP_ROLES, LOCKED_BASELINE_COLUMNS


@dataclass(frozen=True)
class Fold:
    name: str
    start: str
    end: str
    inner_start: str


FOLDS = (
    Fold("summer", "2025-07-01", "2026-01-01", "2025-04-01"),
    Fold("q1", "2026-01-01", "2026-04-01", "2025-07-01"),
    Fold("spring", "2026-04-01", "2026-06-01", "2026-01-01"),
    Fold("midseason", "2026-06-01", "2026-08-09", "2026-04-01"),
)


def _vocabulary(values: pd.Series) -> dict[str, int]:
    return {value: index + 1 for index, value in enumerate(sorted(set(values.astype(str))))}


def _encode(values: pd.Series, vocabulary: dict[str, int]) -> np.ndarray:
    return values.astype(str).map(vocabulary).fillna(0).astype("int64").to_numpy()


def _arrays(frame: pd.DataFrame, train: pd.DataFrame) -> dict[str, np.ndarray]:
    team_values = pd.concat(
        [train["category_blue_team_id"], train["category_red_team_id"]]
    )
    player_columns = [
        f"category_{side}_player_id_{role}"
        for side in ("blue", "red")
        for role in LINEUP_ROLES
    ]
    champion_columns = [
        f"category_{side}_champion_{role}"
        for side in ("blue", "red")
        for role in LINEUP_ROLES
    ]
    team_vocab = _vocabulary(team_values)
    player_vocab = _vocabulary(pd.concat([train[column] for column in player_columns]))
    champion_vocab = _vocabulary(
        pd.concat([train[column] for column in champion_columns])
    )
    league_vocab = _vocabulary(train["category_league"])
    patch_vocab = _vocabulary(train["category_source_patch"])
    numeric = frame[list(LOCKED_BASELINE_COLUMNS)].astype(float).to_numpy("float32")
    train_numeric = train[list(LOCKED_BASELINE_COLUMNS)].astype(float).to_numpy("float32")
    mean = train_numeric.mean(axis=0)
    scale = train_numeric.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return {
        "blue_team": _encode(frame["category_blue_team_id"], team_vocab),
        "red_team": _encode(frame["category_red_team_id"], team_vocab),
        "blue_players": np.column_stack(
            [_encode(frame[f"category_blue_player_id_{role}"], player_vocab) for role in LINEUP_ROLES]
        ),
        "red_players": np.column_stack(
            [_encode(frame[f"category_red_player_id_{role}"], player_vocab) for role in LINEUP_ROLES]
        ),
        "blue_champions": np.column_stack(
            [_encode(frame[f"category_blue_champion_{role}"], champion_vocab) for role in LINEUP_ROLES]
        ),
        "red_champions": np.column_stack(
            [_encode(frame[f"category_red_champion_{role}"], champion_vocab) for role in LINEUP_ROLES]
        ),
        "league": _encode(frame["category_league"], league_vocab),
        "patch": _encode(frame["category_source_patch"], patch_vocab),
        "numeric": ((numeric - mean) / scale).astype("float32"),
        "y": frame["y"].astype("float32").to_numpy(),
        "sizes": np.asarray(
            [len(team_vocab) + 1, len(player_vocab) + 1, len(champion_vocab) + 1, len(league_vocab) + 1, len(patch_vocab) + 1],
            dtype="int64",
        ),
    }


def _train_predict(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(23071)
    torch.set_num_threads(max(1, (torch.get_num_threads())))
    all_frame = pd.concat([train, validation, test], ignore_index=True)
    all_arrays = _arrays(all_frame, train)
    n_train = len(train)
    n_validation = len(validation)

    def take(start: int, end: int) -> dict[str, np.ndarray]:
        return {key: value[start:end] for key, value in all_arrays.items() if key != "sizes"}

    train_arrays = take(0, n_train)
    validation_arrays = take(n_train, n_train + n_validation)
    test_arrays = take(n_train + n_validation, len(all_frame))
    sizes = all_arrays["sizes"]

    class GraphModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            dim = 16
            self.team = nn.Embedding(int(sizes[0]), dim, padding_idx=0)
            self.player = nn.Embedding(int(sizes[1]), dim, padding_idx=0)
            self.champion = nn.Embedding(int(sizes[2]), dim, padding_idx=0)
            self.league = nn.Embedding(int(sizes[3]), 4, padding_idx=0)
            self.patch = nn.Embedding(int(sizes[4]), 4, padding_idx=0)
            signed = dim * 6 + len(LOCKED_BASELINE_COLUMNS)
            self.linear = nn.Linear(signed, 1, bias=False)
            self.hidden = nn.Sequential(
                nn.Linear(signed + 8, 96),
                nn.SiLU(),
                nn.Dropout(0.10),
                nn.Linear(96, 32),
                nn.SiLU(),
                nn.Linear(32, 1, bias=False),
            )

        def side(self, team: torch.Tensor, players: torch.Tensor, champions: torch.Tensor) -> tuple[torch.Tensor, ...]:
            team_value = self.team(team)
            player_values = self.player(players)
            champion_values = self.champion(champions)
            player_mean = player_values.mean(dim=1)
            champion_mean = champion_values.mean(dim=1)
            player_champion = (player_values * champion_values).mean(dim=1)
            team_player = team_value * player_mean
            champion_synergy = (
                champion_values.sum(dim=1).square()
                - champion_values.square().sum(dim=1)
            ) / 20.0
            return team_value, player_mean, champion_mean, player_champion, team_player, champion_synergy

        def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            blue = self.side(batch["blue_team"], batch["blue_players"], batch["blue_champions"])
            red = self.side(batch["red_team"], batch["red_players"], batch["red_champions"])
            signed = torch.cat([*(left - right for left, right in zip(blue, red)), batch["numeric"]], dim=1)
            context = torch.cat([self.league(batch["league"]), self.patch(batch["patch"])], dim=1)
            direct = self.hidden(torch.cat([signed, context], dim=1))
            reverse = self.hidden(torch.cat([-signed, context], dim=1))
            return (self.linear(signed) + 0.5 * (direct - reverse)).squeeze(1)

    def tensors(values: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        result = {}
        for key, value in values.items():
            dtype = torch.float32 if key in {"numeric", "y"} else torch.long
            result[key] = torch.as_tensor(value, dtype=dtype)
        return result

    tr = tensors(train_arrays)
    va = tensors(validation_arrays)
    te = tensors(test_arrays)

    def fit(model: nn.Module, values: dict[str, torch.Tensor], epochs: int, evaluate: dict[str, torch.Tensor] | None = None) -> tuple[nn.Module, int]:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
        loss_fn = nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(23071)
        best_state = copy.deepcopy(model.state_dict())
        best_auc = -1.0
        best_epoch = epochs
        patience = 24
        stale = 0
        for epoch in range(epochs):
            model.train()
            for indices in torch.randperm(len(values["y"]), generator=generator).split(256):
                optimizer.zero_grad(set_to_none=True)
                batch = {key: value[indices] for key, value in values.items() if key != "y"}
                loss = loss_fn(model(batch), values["y"][indices])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            if evaluate is not None:
                model.eval()
                with torch.no_grad():
                    probability = torch.sigmoid(model({key: value for key, value in evaluate.items() if key != "y"})).numpy()
                auc = roc_auc_score(evaluate["y"].numpy(), probability)
                if auc > best_auc + 1e-5:
                    best_auc = auc
                    best_epoch = epoch + 1
                    best_state = copy.deepcopy(model.state_dict())
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break
        if evaluate is not None:
            model.load_state_dict(best_state)
        return model, best_epoch

    model, best_epoch = fit(GraphModel(), tr, 240, va)
    combined = {key: torch.cat([tr[key], va[key]]) for key in tr}
    final, _ = fit(GraphModel(), combined, best_epoch, None)
    final.eval()
    with torch.no_grad():
        return torch.sigmoid(final({key: value for key, value in te.items() if key != "y"})).numpy()


def run(matrix: Path, output: Path) -> None:
    frame = pd.read_parquet(matrix)
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    bounds = frame.groupby("series_id", sort=False)["date"].agg(["min", "max"])
    outputs = []
    for index, fold in enumerate(FOLDS):
        start = pd.Timestamp(fold.start, tz="UTC")
        end = pd.Timestamp(fold.end, tz="UTC")
        inner_start = pd.Timestamp(fold.inner_start, tz="UTC")
        training_ids = bounds[bounds["max"].lt(inner_start)].index
        validation_ids = bounds[bounds["min"].ge(inner_start) & bounds["max"].lt(start)].index
        test_ids = bounds[bounds["min"].ge(start) & bounds["max"].lt(end)].index
        training = frame[frame["series_id"].isin(training_ids)].copy()
        validation = frame[frame["series_id"].isin(validation_ids)].copy()
        test = frame[frame["series_id"].isin(test_ids)].copy()
        probability = _train_predict(training, validation, test)
        result = test[["game_uid", "series_id", "date", "league", "source_patch", "y"]].copy()
        result["p"] = probability
        result["fold"] = index
        outputs.append(result)
        print(fold.name, len(training), len(validation), len(test), roc_auc_score(test["y"], probability), flush=True)
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_parquet(output, index=False)
    print("pooled", len(combined), roc_auc_score(combined["y"], combined["p"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.matrix, args.output)


if __name__ == "__main__":
    main()
