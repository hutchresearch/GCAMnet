from pathlib import Path


def main_queries(xml_path: Path, data_path: Path):
    """Expoloratory analysis on the available xpath queries to extract gcam databases."""
    import pandas as pd
    from lxml import etree

    parser = etree.XMLParser(ns_clean=True, recover=True)
    doc = etree.parse(xml_path, parser)

    queries = []
    for q in doc.xpath("//*[@title]"):
        subgroup = q.xpath("..")[0]
        group = subgroup.xpath("..")[0]
        queries.append(
            {
                "group": group.get("name"),
                "subgroup": subgroup.get("name"),
                "query": q.get("title"),
            },
        )
    queries = pd.DataFrame(queries)
    queries["name"] = queries["query"].str.replace(" ", "_").str.lower()

    data = []
    for f in data_path.iterdir():
        with open(f, "r") as fp:
            data.append(
                {"name": f.stem, "cols": sorted(fp.readline().strip().split("|"))},
            )
    merged = pd.merge(queries, pd.DataFrame(data), on="name")
    return merged
