"""Topic labels shared by the Part 2 scripts: head / mid / tail category groups."""
import pandas as pd

import config

GROUPS = ("head", "mid", "tail")


def group_of(n_train_docs: int) -> str:
    if n_train_docs >= config.HEAD_MIN_DOCS:
        return "head"
    if n_train_docs >= config.MID_MIN_DOCS:
        return "mid"
    return "tail"


def category_groups(train_primary: pd.Series) -> pd.Series:
    """Map each primary category to head / mid / tail using train counts only.

    Categories that never occur in train (possible in val / test) are tail.
    """
    return train_primary.value_counts().map(group_of)


def assign_groups(primary: pd.Series, cat_to_group: pd.Series) -> pd.Series:
    return primary.map(cat_to_group).fillna("tail")
