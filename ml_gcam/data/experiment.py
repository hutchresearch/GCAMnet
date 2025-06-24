def experiment_name_to_label(name) -> str:
    """Convert key to titlecase."""
    m = {
        "binary": "binary",
        "interp_hypercube": "hypercube",
        "interp_dgsm": "dgsm",
        "mixed": "mixed",
        "core": "gcam",
    }
    return m[name]


def experiment_label_to_name(label) -> str:
    """Convert titlecase to key."""
    m = {
        "binary": "binary",
        "hypercube": "interp_hypercube",
        "dgsm": "interp_dgsm",
        "gcam": "core",
    }
    return m[label]


def experiment_name_to_paper_label(name) -> str:
    """Convert key to titlecase."""
    m = {
        "binary": "binary",
        "interp_hypercube": "interpolated",
        "interp_random": "interpolated",
        "interp_dgsm": "interpolated",
        "mixed": "mixed",
        "core": "GCAM",
    }
    return m[name]

def region_to_continent(name) -> str:
    m = {
        "Africa_Eastern": "Africa",
        "Africa_Northern": "Africa",
        "Africa_Southern": "Africa",
        "Africa_Western": "Africa",
        "Argentina": "South America",
        "Australia_NZ": "Australia",
        "Brazil": "South America",
        "Canada": "North America",
        "Central America and Caribbean": "North America",
        "Central Asia": "Asia",
        "China": "Asia",
        "Colombia": "South America",
        "EU-12" : "Europe",
        "EU-15" : "Europe",
        "Europe_Eastern" : "Europe",
        "Europe_Non_EU" : "Europe",
        "European Free Trade Association" : "Europe",
        "India" : "Asia",
        "Indonesia" : "Asia",
        "Japan" : "Asia",
        "Mexico" : "North America",
        "Middle East" : "Asia",
        "Pakistan" : "Asia",
        "Russia" : "Asia",
        "South Africa" : "Africa",
        "South America_Northern" : "South America",
        "South America_Southern" : "South America",
        "South Asia" : "Asia",
        "South Korea" : "Asia",
        "Southeast Asia" : "Asia",
        "Taiwan" : "Asia",
        "USA" : "North America",
    }

    return m[name]

def quantity_to_sector(name) -> str:
    return name.split("_")[0]
