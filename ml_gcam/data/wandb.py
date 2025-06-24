import pandas as pd

import wandb

from .. import config


def history(group_name=None, run_id=None):
    """Download a wandb run from the api."""
    api = wandb.Api()
    if group_name is not None:
        filters = {"group": group_name}
        runs = api.runs(
            path=config.wandb.entity + "/" + config.project,
            filters=filters,
        )
    elif run_id is not None:
        runs = [api.run(f"{config.wandb.entity}/{config.project}/{run_id}")]
    else:
        raise ValueError("one or the other has to be set: group_name, run_id")
    collect = []
    for run in runs:
        history = run.history()
        collect.append(history)
    df = pd.concat(collect, ignore_index=True)
    return df
