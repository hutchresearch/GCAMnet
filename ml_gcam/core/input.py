import os
import re
from pathlib import Path

import pandas as pd
from lxml import etree


def main(output_dir: Path, output_file: Path, config_dir: Path):
    """Parse scenario config -> inputs used."""
    # inputs = Path(GCAM_CORE_PATH) / "exp1_jr_file/inputs"
    config_dir = Path(config_dir)
    rows = []
    # get all the xml configs in the dir
    configs = [
        x
        for x in config_dir.iterdir()
        if x.name.endswith("xml") and x.stat().st_size != 0
    ]
    for f in configs:
        row = {}
        root = etree.parse(f)
        row["filename"] = f
        row["scenario_name"] = root.find('//Strings/Value[@name = "scenarioName"]').text
        values = root.findall("//ScenarioComponents/Value")
        for el in values:
            row[el.get("name")] = el.text
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / output_file, index=False, sep="|")


def parse(batch_path: Path, filename, filedir):
    """Exploratory analysis of a single config file from the experiment."""
    doc = etree.parse(batch_path)
    root = doc.getroot()
    sets = root.xpath("./ComponentSet")

    runs = []
    for run in sets:
        category = run.get("name").lower().replace(" ", "_")
        subset = run.xpath("./FileSet")
        for s in subset:
            label = s.get("name")
            values = s.xpath("./Value")
            if values:
                for val in values:
                    metric = val.get("name")
                    filename = os.path.split(val.text)[1]
                    runs.append(
                        {
                            "category": category,
                            "label": label,
                            "metric": metric,
                            "filename": filename,
                        },
                    )
            else:
                runs.append(
                    {
                        "category": category,
                        "label": label,
                        "metric": None,
                        "filename": None,
                    },
                )
    df_runs = pd.DataFrame(runs)
    return df_runs


def defaults(config_file: Path):
    """Finds the common default values in the config file of a gcam run."""
    doc = etree.parse(config_file)
    root = doc.getroot()

    defaults = []
    for x in root.xpath("//ScenarioComponents/Value"):
        metric = x.get("name")
        filename = os.path.split(x.text)[1]
        defaults.append({"metric": metric, "filename": filename})
    return pd.DataFrame(defaults).sort_values("metric")
