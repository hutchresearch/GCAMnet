import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import polars as pl
from polars.exceptions import PolarsError

from .. import config, logger

OUTPUT_INDEX = ["experiment", "scenario", "region", "year"]


def extract_outputs_from_gcamreader(
    experiment,
    queries,
    outputs_path: Path,
):
    """Aggregated raw extract csv from an experiments."""
    functions = {
        "energy_demand_share_primary": energy_demand_share_primary,
        "energy_demand_share_electricity": energy_demand_share_electricity,
        "emissions_capture": emissions_capture,
        "energy_prices": energy_prices,
        "energy_supply_share_electricity": energy_supply_share_electricity,
        "energy_supply_share_primary": energy_supply_share_primary,
        "land_demand": land_demand,
        "land_prices": land_prices,
        "land_supply_allocation": land_supply_allocation,
        "land_supply_production": land_supply_production,
        "water_demand": water_demand,
        "agriculture_prices": agriculture_prices,
        "water_consumption": water_consumption,
        "electricity_supply": electricity_supply,
    }
    start_time = time.perf_counter()
    extractors = []
    for query in queries:
        logger.info(f"finding gcamreader files for {experiment}: {query}")
        if query in functions:
            extractor = functions[query](experiment, outputs_path)
            if extractor.inputs_found == 0:
                logger.warning(f"{query} return no results. skipping")
            else:
                logger.info(
                    f"found {extractor.inputs_found} {extractor.filename} files for {query}",
                )
                extractors.append(extractor)
        else:
            logger.error(f"unrecognized query {query}")

    if bool(int(config.pretend)):
        return
    if len(extractors) == 0:
        logger.error(f"all {experiment} queries failed {queries}", exc_info=True)
        return
    df = None
    for extractor in extractors:
        result = extractor.execute()
        if result is None:
            logger.error(f"{query} return no results. skipping")
            continue
        else:
            if df is None:
                df = result
            else:
                try:
                    df = df.join(result, on=OUTPUT_INDEX, how="outer_coalesce")
                except PolarsError:
                    logger.error(f"failed to join {query}", exc_info=True)
                    continue

    df = df.fill_null(0)
    end_time = time.perf_counter()
    logger.info(f"done with all queries [{end_time - start_time:.2f} seconds]")

    return df


@dataclass
class Extractor:
    """wraps a lot of similar functionality for extracting and aggregating gcam data."""

    experiment: str
    outputs_path: Path
    filename: str
    index_columns: List[str] = field(default_factory=lambda: OUTPUT_INDEX)
    columns: List = field(init=False, default_factory=list)
    agg: List = field(init=False, default_factory=list)
    inputs: List = field(init=False, default_factory=list)
    inputs_found: int = 0

    def __post_init__(self):
        self.columns = [
            pl.lit(self.experiment).alias("experiment").cast(pl.String),
            pl.col("scenario").str.replace(",.*", "").cast(pl.String).alias("scenario"),
            pl.col("region").cast(pl.String).alias("region"),
            pl.col("Year").cast(pl.UInt16).alias("year"),
            pl.col("value").cast(pl.Float64).alias("value"),
        ]
        self.inputs = list(self.outputs_path.glob(f"*/{self.filename}"))
        self.inputs_found = len(self.inputs)

    def add_columns(self, columns):
        self.columns.extend(columns)

    def add_agg(self, agg):
        self.agg.extend(agg)

    def queries(self):
        for f in self.inputs:
            result = (
                pl.scan_csv(f, separator="|")
                .select(*self.columns)
                .groupby(self.index_columns)
                .agg(*self.agg)
                .fill_null(0)
            )
            yield result

    def execute(self):
        if self.inputs_found == 0:
            logger.error(f"no files name {self.filename} found")
            return None
        logger.info(f"processing {self.inputs_found} {self.filename} files into memory")
        try:
            start_time = time.perf_counter()
            result = pl.collect_all(self.queries())
            result = pl.concat(result)
            results_found = len(result)
            end_time = time.perf_counter()
            logger.info(
                f"done processing {results_found:,} rows from {self.inputs_found:,} {self.filename} files for {self.experiment} [{end_time - start_time:.2f} seconds]",
            )
            return result
        except PolarsError:
            logger.error(f"extract execute {self.filename} failed", exc_info=True)

            for c in self.columns:
                logger.error(f"col: {c}")

            for a in self.agg:
                logger.error(f"agg: {a}")
            return None


def energy_demand_share_primary(experiment, outputs_path) -> Extractor:
    biomass_input = ["biomass"]
    fuel_input = ["gas", "coal", "refined liquids"]
    transport_sectors = [
        "H2 forecourt production",
        "trn_freight",
        "trn_freight_road",
        "trn_pass",
        "trn_pass_road",
        "trn_pass_road_LDV",
        "trn_pass_road_LDV_4W",
    ]
    industry_sectors = [
        "H2 central production",
        "cement",
        "desalinated water",
        "industrial energy use",
        "industrial wastewater treatment",
        "industrial water abstraction",
        "industrial water treatment",
        "irrigation water abstraction",
        "refining",
    ]
    building_sectors = [
        "comm cooling",
        "comm others",
        "municipal wastewater treatment",
        "municipal water abstraction",
        "municipal water distribution",
        "municipal water treatment",
        "resid cooling",
        "resid others",
        "comm heating",
        "resid heating",
    ]

    extractor = Extractor(
        experiment,
        outputs_path,
        "final_energy_consumption_by_sector_and_fuel.csv",
    )
    extractor.add_columns(
        [
            pl.col("input").cast(pl.String),
            pl.col("sector").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("input") == "hydrogen")
            .sum()
            .alias("energy_demand_fuel_hydrogen"),
            pl.col("value")
            .filter(
                (pl.col("input").is_in(fuel_input))
                & (pl.col("sector").is_in(transport_sectors)),
            )
            .sum()
            .alias("energy_demand_fuel_fossil_transport"),
            pl.col("value")
            .filter(
                (pl.col("input").is_in(fuel_input))
                & (pl.col("sector").is_in(industry_sectors)),
            )
            .sum()
            .alias("energy_demand_fuel_fossil_industry"),
            pl.col("value")
            .filter(
                (pl.col("input").is_in(fuel_input))
                & (pl.col("sector").is_in(building_sectors)),
            )
            .sum()
            .alias("energy_demand_fuel_fossil_building"),
            pl.col("value")
            .filter(
                (pl.col("input").is_in(biomass_input))
                & (pl.col("sector").is_in(transport_sectors)),
            )
            .sum()
            .alias("energy_demand_fuel_biomass_transport"),
            pl.col("value")
            .filter(
                (pl.col("input").is_in(biomass_input))
                & (pl.col("sector").is_in(industry_sectors)),
            )
            .sum()
            .alias("energy_demand_fuel_biomass_industry"),
            pl.col("value")
            .filter(
                (pl.col("input").is_in(biomass_input))
                & (pl.col("sector").is_in(building_sectors)),
            )
            .sum()
            .alias("energy_demand_fuel_biomass_building"),
        ],
    )
    return extractor


def energy_demand_share_electricity(experiment, outputs_path) -> Extractor:
    extractor = Extractor(
        experiment,
        outputs_path,
        "elec_consumption_by_demand_sector.csv",
    )
    extractor.add_columns(
        [
            pl.col("input").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("input") == "elect_td_trn")
            .sum()
            .alias("energy_demand_elec_transport"),
            pl.col("value")
            .filter(pl.col("input") == "elect_td_ind")
            .sum()
            .alias("energy_demand_elec_industry"),
            pl.col("value")
            .filter(pl.col("input") == "elect_td_bld")
            .sum()
            .alias("energy_demand_elec_building"),
        ],
    )
    return extractor


def emissions_capture(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "co2_sequestration_by_tech.csv")
    extractor.add_agg([pl.col("value").sum().alias("emission_capture_ccs")])
    return extractor


def energy_prices(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "final_energy_prices.csv")
    extractor.add_columns(
        [
            pl.col("fuel").cast(pl.String),
        ],
    )

    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("fuel").str.starts_with("elect"))
            .mean()
            .alias("energy_price_electricity"),
            pl.col("value")
            .filter(pl.col("fuel") == "delivered coal")
            .mean()
            .alias("energy_price_coal"),
            pl.col("value")
            .filter(pl.col("fuel").str.ends_with("gas"))
            .mean()
            .alias("energy_price_gas"),
            pl.col("value")
            .filter(pl.col("fuel").str.contains("liquids"))
            .mean()
            .alias("energy_price_oil"),
        ],
    )
    return extractor


def energy_supply_share_electricity(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "elec_gen_by_subsector.csv")
    extractor.add_columns(
        [
            pl.col("subsector").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("subsector") == "coal")
            .sum()
            .alias("energy_supply_electricity_coal"),
            pl.col("value")
            .filter(pl.col("subsector") == "refined liquids")
            .sum()
            .alias("energy_supply_electricity_oil"),
            pl.col("value")
            .filter(pl.col("subsector") == "gas")
            .sum()
            .alias("energy_supply_electricity_gas"),
            pl.col("value")
            .filter(pl.col("subsector") == "solar")
            .sum()
            .alias("energy_supply_electricity_solar"),
            pl.col("value")
            .filter(pl.col("subsector") == "wind")
            .sum()
            .alias("energy_supply_electricity_wind"),
            pl.col("value")
            .filter(pl.col("subsector") == "biomass")
            .sum()
            .alias("energy_supply_electricity_biomass"),
            pl.col("value")
            .filter(pl.col("subsector") == "nuclear")
            .sum()
            .alias("energy_supply_electricity_nuclear"),
            pl.col("value")
            .filter(
                ~pl.col("subsector").is_in(
                    [
                        "coal",
                        "refined liquids",
                        "gas",
                        "solar",
                        "wind",
                        "biomass",
                        "nuclear",
                    ],
                ),
            )
            .sum()
            .alias("energy_supply_electricity_other"),
        ],
    )
    return extractor


def energy_supply_share_primary(experiment, outputs_path) -> Extractor:
    extractor = Extractor(
        experiment,
        outputs_path,
        "primary_energy_consumption_by_region_(direct_equivalent).csv",
    )
    extractor.add_columns(
        [
            pl.col("fuel").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("fuel") == "c coal")
            .sum()
            .alias("energy_supply_primary_coal"),
            pl.col("value")
            .filter(pl.col("fuel") == "a oil")
            .sum()
            .alias("energy_supply_primary_oil"),
            pl.col("value")
            .filter(pl.col("fuel") == "b natural gas")
            .sum()
            .alias("energy_supply_primary_gas"),
            pl.col("value")
            .filter(pl.col("fuel") == "h solar")
            .sum()
            .alias("energy_supply_primary_solar"),
            pl.col("value")
            .filter(pl.col("fuel") == "g wind")
            .sum()
            .alias("energy_supply_primary_wind"),
            pl.col("value")
            .filter(pl.col("fuel") == "d biomass")
            .sum()
            .alias("energy_supply_primary_biomass"),
            pl.col("value")
            .filter(pl.col("fuel") == "e nuclear")
            .sum()
            .alias("energy_supply_primary_nuclear"),
            pl.col("value")
            .filter(
                ~pl.col("fuel").is_in(
                    [
                        "d biomass",
                        "g wind",
                        "h solar",
                        "b natural gas",
                        "a oil",
                        "e nuclear",
                        "c coal",
                    ],
                ),
            )
            .sum()
            .alias("energy_supply_primary_other"),
        ],
    )
    return extractor


def land_demand(experiment, outputs_path) -> Extractor:
    extractor = Extractor(
        experiment,
        outputs_path,
        "demand_balances_by_crop_commodity.csv",
    )
    extractor.add_columns(
        [
            pl.col("sector").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("sector").str.starts_with("Feed"))
            .sum()
            .alias("land_demand_feed"),
            pl.col("value")
            .filter(pl.col("sector").str.starts_with("Food"))
            .sum()
            .alias("land_demand_food"),
        ],
    )
    return extractor


def land_prices(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "prices_by_sector.csv")
    extractor.add_columns(
        [
            pl.col("sector").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("sector").str.contains("biomass"))
            .mean()
            .alias("land_price_biomass"),
            pl.col("value")
            .filter(pl.col("sector").str.contains("forest"))
            .mean()
            .alias("land_price_forest"),
        ],
    )
    return extractor


def land_supply_allocation(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "aggregated_land_allocation.csv")
    extractor.add_columns(
        [
            pl.col("LandLeaf").cast(pl.String).alias("landleaf"),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("landleaf").str.starts_with("forest"))
            .sum()
            .alias("land_allocation_forest"),
            pl.col("value")
            .filter(pl.col("landleaf") == "biomass")
            .sum()
            .alias("land_allocation_biomass"),
            pl.col("value")
            .filter(pl.col("landleaf").str.starts_with("pasture"))
            .sum()
            .alias("land_allocation_pasture"),
            pl.col("value")
            .filter(pl.col("landleaf").is_in(["grass", "shrub"]))
            .sum()
            .alias("land_allocation_grass_shrub"),
            pl.col("value")
            .filter(
                ~pl.col("landleaf").str.starts_with("forest")
                & ~(pl.col("landleaf") == "biomass")
                & ~pl.col("landleaf").str.starts_with("pasture")
                & ~pl.col("landleaf").is_in(["grass", "shrub"]),
            )
            .sum()
            .alias("land_allocation_other"),
            pl.col("value").sum().alias("land_allocation_all"),
        ],
    )
    return extractor


def land_supply_production(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "ag_production_by_crop_type.csv")
    extractor.add_columns([pl.col("output").cast(pl.String).alias("output")])
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("output") == "Forest")
            .sum()
            .alias("land_production_forest"),
            pl.col("value")
            .filter(pl.col("output") == "biomass")
            .sum()
            .alias("land_production_biomass"),
            pl.col("value")
            .filter(pl.col("output") == "Pasture")
            .sum()
            .alias("land_production_pasture"),
            pl.col("value")
            .filter(pl.col("output") == "FodderGrass")
            .sum()
            .alias("land_production_grass_shrub"),
            pl.col("value")
            .filter(
                ~pl.col("output").is_in(
                    ["Forest", "biomass", "Pasture", "FodderGrass"],
                ),
            )
            .sum()
            .alias("land_production_other"),
        ],
    )
    return extractor


def water_demand(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "water_withdrawals_by_tech.csv")
    extractor.add_columns(
        [
            pl.col("technology").cast(pl.String),
            pl.col("sector").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("sector") == "electricity")
            .sum()
            .alias("water_demand_electricity"),
            pl.col("value")
            .filter(pl.col("technology").str.contains("IRR"))
            .sum()
            .alias("water_demand_crops"),
        ],
    )
    return extractor


def agriculture_prices(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "ag_commodity_prices.csv")
    extractor.add_columns(
        [
            pl.col("sector").cast(pl.String).alias("sector"),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("sector") == "Corn")
            .sum()
            .alias("agriculture_prices_rice"),
            pl.col("value")
            .filter(pl.col("sector") == "Rice")
            .sum()
            .alias("agriculture_prices_corn"),
            pl.col("value")
            .filter(pl.col("sector") == "Wheat")
            .sum()
            .alias("agriculture_prices_wheat"),
            pl.col("value")
            .filter(pl.col("sector") == "OilCrop")
            .sum()
            .alias("agriculture_prices_oilcrop"),
        ],
    )
    return extractor


def water_consumption(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "water_consumption_by_region.csv")
    extractor.add_agg([pl.col("value").sum().alias("water_consumption")])
    return extractor


def electricity_supply(experiment, outputs_path) -> Extractor:
    extractor = Extractor(experiment, outputs_path, "elec_gen_by_gen_tech.csv")
    extractor.add_columns(
        [
            pl.col("subsector").cast(pl.String),
        ],
    )
    extractor.add_agg(
        [
            pl.col("value")
            .filter(pl.col("subsector") == "solar")
            .sum()
            .alias("electricity_supply_solar"),
            pl.col("value")
            .filter(pl.col("subsector") == "wind")
            .sum()
            .alias("electricity_supply_wind"),
            pl.col("value")
            .filter(pl.col("subsector") == "gas")
            .sum()
            .alias("electricity_supply_gas"),
            pl.col("value")
            .filter(pl.col("subsector") == "hydro")
            .sum()
            .alias("electricity_supply_hydro"),
            pl.col("value")
            .filter(pl.col("subsector") == "refined liquids")
            .sum()
            .alias("electricity_supply_oil"),
            pl.col("value")
            .filter(pl.col("subsector") == "coal")
            .sum()
            .alias("electricity_supply_coal"),
            pl.col("value")
            .filter(pl.col("subsector") == "biomass")
            .sum()
            .alias("electricity_supply_biomass"),
            pl.col("value")
            .filter(~pl.col("subsector").is_in(["solar", "wind"]))
            .sum()
            .alias("electricity_supply_non_wind_solar"),
        ],
    )
    return extractor
