from pathlib import Path

from config import (
    ConfigurationSet,
    config_from_dict,
    config_from_env,
    config_from_toml,
)
from dotenv import find_dotenv, load_dotenv


def init_config():
    config_path = Path(__file__).parent
    dot = find_dotenv(raise_error_if_not_found=True)
    load_dotenv(dot)
    env = config_from_env(prefix="ML_GCAM", separator="__", lowercase_keys=True)
    toml = config_from_toml(str(config_path / "config.toml"), read_from_file=True)

    patched = {
        "data": {
            "extract_sources": [],
            "dev_sources": [],
            "train_sources": [],
            "input_keys": [],
            "output_keys": [],
            "region_keys": [],
            "year_keys": [],
            "n_dimensions": None,
        },
    }

    for k in toml.data.sources:
        key = toml.data.sources[k]
        if not key.enabled:
            continue
        if key.dev:
            patched["data"]["dev_sources"].append(k)
        if key.train:
            patched["data"]["train_sources"].append(k)
        if key.new_samples:
            patched["data"]["extract_sources"].append(k)

    for o in toml.data.outputs:
        key = toml.data.outputs[o]
        if not key.enabled:
            continue
        patched["data"]["output_keys"].append(key["key"])

    for i in toml.data.inputs:
        key = toml.data.inputs[i]
        patched["data"]["input_keys"].append(key["key"])

    for r in toml.data.regions:
        key = toml.data.regions[r]
        patched["data"]["region_keys"].append(key["key"])

    for y in toml.data.years:
        key = toml.data.years[y]
        patched["data"]["year_keys"].append(int(key["key"]))

    patched["data"]["n_dimensions"] = len(toml.data.regions) * len(toml.data.years)
    patched["data"]["input_keys"].sort()
    patched["data"]["output_keys"].sort()
    patched["data"]["region_keys"].sort()
    patched["data"]["year_keys"].sort()

    patch = config_from_dict(patched)
    configs = [env, toml, patch]

    return ConfigurationSet(*configs)


config = init_config()
