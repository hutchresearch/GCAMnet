from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from xml.dom import minidom

import numpy as np
import pandas as pd
from lxml import etree
from tqdm import tqdm

from .. import config


def sample_bits(samples: int, name: str, save_to: Optional[Path] = None):
    """Creates a metadata.csv with [--sample] samples of input space."""
    if name == "interp_hypercube":
        from scipy.stats import qmc

        columns = config.data.input_keys
        sampler = qmc.LatinHypercube(d=len(columns)-3) # sample in R^9
        bits = sampler.random(n=samples)
        ints = np.random.randint(0, 2, size=(samples, 3))
        for i, j in enumerate([1, 3, 4]):
            bits = np.insert(bits, j, ints[:, i], axis=1) # insert binary
        bits = np.round(bits, 4)
    elif name == "interp_dgsm":
        from scipy.stats import qmc
        from SALib.sample.finite_diff import sample
        import itertools

        columns = config.data.input_keys
        columns.remove('bio')
        columns.remove('elec')
        columns.remove('emiss')
        
        problem = {
            "num_vars": len(columns),
            "names": columns,
            "bounds": [[0, 1]] * len(columns),
        }
        D = problem["num_vars"]
        salib_samples = samples // (D+1)

        bits = sample(problem, salib_samples) # SALib
        ints = list(itertools.product([0,1], repeat=3))
        collect = []
        for combo in ints:
            chunk = np.tile(np.array(combo), (samples,1))
            collect.append(chunk)
        ints = np.vstack(collect)
        bits = np.tile(bits, (8, 1))
        for i, j in enumerate([1, 3, 4]):
            bits = np.insert(bits, j, ints[:, i], axis=1) # insert binary
        bits = np.round(bits, 4)
    elif name == "binary":
        from itertools import product

        bits = list(product([0, 1], repeat=12))
        bits = bits[:samples]
    else:
        print(f"sampling strategy not implemented: {name}")
    df = pd.DataFrame(bits, columns=config.data.input_keys)
    df.reset_index(inplace=True, names="scenario_id")
    if save_to is not None:
        df.to_csv(save_to, sep="|", index=False)
    return bits


@dataclass
class XMLPaths:
    TF2025: str
    BIO: str
    BSSP1E: str
    BSSP1: str
    BSSP5E: str
    BSSP5: str
    CCSHI: str
    CCSLO: str
    CEMSSP1: str
    CEMSSP5: str
    INDSSP1E: str
    INDSSP1: str
    INDSSP5E: str
    INDSSP5: str
    NUCH: str
    NUCL: str
    RECSSP1: str
    RECSSP5: str
    SOCSSP1: str
    SOCSSP5: str
    SOLHTLS: str
    SOLLTHS: str
    SOLA: str
    SOLL: str
    SOLSH: str
    SOLSL: str
    SOLTH: str
    SOLTL: str
    SPA14: str
    TRANSSP1: str
    TRANSSP5: str
    TRANSSP1E: str
    TRANSSP5E: str
    WINDA: str
    WINDHTLS: str
    WINDLTHS: str
    WINDL: str
    WSBH: str
    WSBL: str


PATHS = XMLPaths(
    TF2025="2025_target_finder.xml",
    BIO="bio_constraint_100_hi.xml",
    BSSP1E="building_SSP1_hi_elec.xml",
    BSSP1="building_SSP1.xml",
    BSSP5E="building_SSP5_hi_elec.xml",
    BSSP5="building_SSP5.xml",
    CCSHI="ccs_supply_high.xml",
    CCSLO="ccs_supply_lowest.xml",
    CEMSSP1="cement_incelas_gssp1.xml",
    CEMSSP5="cement_incelas_gssp5.xml",
    INDSSP1E="industry_incelas_gssp1_hi_elec.xml",
    INDSSP1="industry_incelas_gssp1.xml",
    INDSSP5E="industry_incelas_gssp5_hi_elec.xml",
    INDSSP5="industry_incelas_gssp5.xml",
    NUCH="nuclear_adv.xml",
    NUCL="nuclear_low.xml",
    RECSSP1="resources_SSP1.xml",
    RECSSP5="resources_SSP5.xml",
    SOCSSP1="socioeconomics_gSSP1.xml",
    SOCSSP5="socioeconomics_gSSP5.xml",
    SOLA="solar_adv.xml",
    SOLHTLS="solar_hi_tech_lo_storage.xml",
    SOLLTHS="solar_lo_tech_hi_storage.xml",
    SOLL="solar_low.xml",
    SOLSH="solar_storage_hi.xml",
    SOLSL="solar_storage_lo.xml",
    SOLTH="solar_tech_hi.xml",
    SOLTL="solar_tech_lo.xml",
    SPA14="spa14_tax.xml",
    TRANSSP1="transportation_UCD_SSP1_with_CORE.xml",
    TRANSSP5="transportation_UCD_SSP5_with_CORE.xml",
    TRANSSP1E="transport_SSP1_hi_elec_with_CORE.xml",
    TRANSSP5E="transport_SSP5_hi_elec_with_CORE.xml",
    WINDA="wind_adv.xml",
    WINDHTLS="wind_hi_tech_lo_storage.xml",
    WINDLTHS="wind_lo_tech_hi_storage.xml",
    WINDL="wind_low.xml",
    WSBH="wind_solar_backups_hi.xml",
    WSBL="wind_solar_backups_low.xml",
)


def interpolate_nuclear(low_xml, high_xml, scale):
    low_technologies_elements = low_xml.getElementsByTagName("technology")
    high_technologies_elements = high_xml.getElementsByTagName("technology")

    for low_technology_element, high_technology_element in zip(
        low_technologies_elements,
        high_technologies_elements,
    ):
        low_technology_name = low_technology_element.attributes["name"].value
        high_technology_name = high_technology_element.attributes["name"].value

        assert (
            low_technology_name == high_technology_name
        ), f"Technology name is different (low: {low_technology_name}, high: { high_technology_name})"

        low_period_elements = low_technology_element.getElementsByTagName("period")
        high_period_elements = high_technology_element.getElementsByTagName("period")
        for low_period_element, high_period_element in zip(
            low_period_elements,
            high_period_elements,
        ):
            low_input_capital_element = low_period_element.getElementsByTagName(
                "input-capital",
            )[0]
            high_input_capital_element = high_period_element.getElementsByTagName(
                "input-capital",
            )[0]

            low_capital_overnight_element = (
                low_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )
            high_capital_overnight_element = (
                high_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )

            low_capital_overnight = int(low_capital_overnight_element.firstChild.data)
            high_capital_overnight = int(high_capital_overnight_element.firstChild.data)

            low_capital_overnight_element.firstChild.data = int(
                (low_capital_overnight * (1 - scale) + high_capital_overnight * scale),
            )

    return low_xml


def interpolate_socioeconomics(low_xml, high_xml, scale):
    low_region_elements = low_xml.getElementsByTagName("region")
    high_region_elements = high_xml.getElementsByTagName("region")

    for low_region_element, high_region_element in zip(
        low_region_elements,
        high_region_elements,
    ):
        low_region = low_region_element.attributes["name"].value
        high_region = high_region_element.attributes["name"].value

        assert (
            low_region == high_region
        ), f"Region name is different (low: {low_region}, high: {high_region})"

        low_population_elements = low_region_element.getElementsByTagName(
            "populationMiniCAM",
        )
        high_population_elements = high_region_element.getElementsByTagName(
            "populationMiniCAM",
        )
        for low_population_element, high_population_element in zip(
            low_population_elements,
            high_population_elements,
        ):
            low_tot_population_element = low_population_element.getElementsByTagName(
                "totalPop",
            )[0]
            high_tot_population_element = high_population_element.getElementsByTagName(
                "totalPop",
            )[0]

            low_tot_population = int(low_tot_population_element.firstChild.data)
            high_tot_population = int(high_tot_population_element.firstChild.data)

            low_tot_population_element.firstChild.data = int(
                (low_tot_population * (1 - scale) + high_tot_population * scale),
            )

        low_labor_productivity_elements = low_region_element.getElementsByTagName(
            "laborproductivity",
        )
        high_labor_productivity_elements = high_region_element.getElementsByTagName(
            "laborproductivity",
        )
        for low_labor_productivity_element, high_labor_productivity_element in zip(
            low_labor_productivity_elements,
            high_labor_productivity_elements,
        ):
            low_labor_productivity_value = float(
                low_labor_productivity_element.firstChild.data,
            )
            high_labor_productivity_value = float(
                high_labor_productivity_element.firstChild.data,
            )

            low_labor_productivity_element.firstChild.data = float(
                (
                    low_labor_productivity_value * (1 - scale)
                    + high_labor_productivity_value * scale
                ),
            )

    return low_xml


def interpolate_resources(low_xml, high_xml, scale):
    low_region_elements = low_xml.getElementsByTagName("region")
    high_region_elements = high_xml.getElementsByTagName("region")

    for low_region_element, high_region_element in zip(
        low_region_elements,
        high_region_elements,
    ):
        low_region = low_region_element.attributes["name"].value
        high_region = high_region_element.attributes["name"].value

        assert (
            low_region == high_region
        ), f"Region name is different (low: {low_region}, high: {high_region})"

        low_resource_elements = low_region_element.getElementsByTagName(
            "reserve-subresource",
        )
        high_resource_elements = high_region_element.getElementsByTagName(
            "reserve-subresource",
        )

        for low_resource_element, high_resource_element in zip(
            low_resource_elements,
            high_resource_elements,
        ):
            low_resouce_name = low_resource_element.attributes["name"].value
            high_resouce_name = high_resource_element.attributes["name"].value

            assert (
                low_resouce_name == high_resouce_name
            ), f"Resource name is different (low: {low_resouce_name}, high: {high_resouce_name})"

            low_period_elements = low_resource_element.getElementsByTagName("period")
            high_period_elements = high_resource_element.getElementsByTagName("period")
            for low_period_element, high_period_element in zip(
                low_period_elements,
                high_period_elements,
            ):
                low_resouce_input_environ_cost_element = (
                    low_period_element.getElementsByTagName("input-cost")[0]
                )
                high_resouce_input_environ_cost_element = (
                    high_period_element.getElementsByTagName("input-cost")[0]
                )

                low_resouce_input_environ_cost = float(
                    low_resouce_input_environ_cost_element.firstChild.data,
                )
                high_resouce_input_environ_cost = float(
                    high_resouce_input_environ_cost_element.firstChild.data,
                )

                low_resouce_input_environ_cost_element.firstChild.data = float(
                    (
                        low_resouce_input_environ_cost * (1 - scale)
                        + high_resouce_input_environ_cost * scale
                    ),
                )

            low_tech_change_elements = low_resource_element.getElementsByTagName(
                "techChange",
            )
            high_tech_change_elements = high_resource_element.getElementsByTagName(
                "techChange",
            )
            for low_tech_change_element, high_tech_change_element in zip(
                low_tech_change_elements,
                high_tech_change_elements,
            ):
                low_tech_change_value = float(low_tech_change_element.firstChild.data)
                high_tech_change_value = float(high_tech_change_element.firstChild.data)

                low_tech_change_element.firstChild.data = float(
                    (
                        low_tech_change_value * (1 - scale)
                        + high_tech_change_value * scale
                    ),
                )

    return low_xml


def interpolate_building(low_xml, high_xml, scale):
    low_region_elements = low_xml.getElementsByTagName("region")
    high_region_elements = high_xml.getElementsByTagName("region")

    for low_region_element, high_region_element in zip(
        low_region_elements,
        high_region_elements,
    ):
        low_region = low_region_element.attributes["name"].value
        high_region = high_region_element.attributes["name"].value

        assert (
            low_region == high_region
        ), f"Region name is different (low: {low_region}, high: {high_region})"

        low_demand_function_elements = low_region_element.getElementsByTagName(
            "satiation-demand-function",
        )
        high_demand_function_elements = high_region_element.getElementsByTagName(
            "satiation-demand-function",
        )

        low_comm_building_satiation_demand_level = float(
            low_demand_function_elements[0]
            .getElementsByTagName("satiation-level")[0]
            .firstChild.data,
        )
        high_comm_building_satiation_demand_level = float(
            high_demand_function_elements[0]
            .getElementsByTagName("satiation-level")[0]
            .firstChild.data,
        )

        low_demand_function_elements[0].getElementsByTagName("satiation-level")[
            0
        ].firstChild.data = float(
            (
                low_comm_building_satiation_demand_level * (1 - scale)
                + high_comm_building_satiation_demand_level * scale
            ),
        )

        low_comm_building_satiation_demand_adder = float(
            low_demand_function_elements[0]
            .getElementsByTagName("satiation-adder")[0]
            .firstChild.data,
        )
        high_comm_building_satiation_demand_adder = float(
            high_demand_function_elements[0]
            .getElementsByTagName("satiation-adder")[0]
            .firstChild.data,
        )

        low_demand_function_elements[0].getElementsByTagName("satiation-adder")[
            0
        ].firstChild.data = float(
            (
                low_comm_building_satiation_demand_adder * (1 - scale)
                + high_comm_building_satiation_demand_adder * scale
            ),
        )

        low_comm_building_satiation_service_demand_level = float(
            low_demand_function_elements[1]
            .getElementsByTagName("satiation-level")[0]
            .firstChild.data,
        )
        high_comm_building_satiation_service_demand_level = float(
            high_demand_function_elements[1]
            .getElementsByTagName("satiation-level")[0]
            .firstChild.data,
        )

        low_demand_function_elements[1].getElementsByTagName("satiation-level")[
            0
        ].firstChild.data = float(
            (
                low_comm_building_satiation_service_demand_level * (1 - scale)
                + high_comm_building_satiation_service_demand_level * scale
            ),
        )

        low_resid_building_satiation_demand_level = float(
            low_demand_function_elements[2]
            .getElementsByTagName("satiation-level")[0]
            .firstChild.data,
        )
        high_resid_building_satiation_demand_level = float(
            high_demand_function_elements[2]
            .getElementsByTagName("satiation-level")[0]
            .firstChild.data,
        )

        low_demand_function_elements[2].getElementsByTagName("satiation-level")[
            0
        ].firstChild.data = float(
            (
                low_resid_building_satiation_demand_level * (1 - scale)
                + high_resid_building_satiation_demand_level * scale
            ),
        )

    return low_xml


def intperpolate_ccs_supply(low_xml, high_xml, scale):
    low_region_elements = low_xml.getElementsByTagName("region")
    high_region_elements = high_xml.getElementsByTagName("region")

    for low_region_element, high_region_element in zip(
        low_region_elements,
        high_region_elements,
    ):
        low_grade_elements = low_region_element.getElementsByTagName("grade")
        high_grade_elements = high_region_element.getElementsByTagName("grade")

        for low_grade_element, high_grade_element in zip(
            low_grade_elements,
            high_grade_elements,
        ):
            low_grade_name = low_grade_element.attributes["name"].value
            high_grade_name = high_grade_element.attributes["name"].value

            assert (
                low_grade_name == high_grade_name
            ), "Low grade name must be same as high grade name"

            low_extraction_cost = float(
                low_grade_element.getElementsByTagName("extractioncost")[
                    0
                ].firstChild.data,
            )
            high_extraction_cost = float(
                high_grade_element.getElementsByTagName("extractioncost")[
                    0
                ].firstChild.data,
            )

            low_extraction_cost_element = low_grade_element.getElementsByTagName(
                "extractioncost",
            )[0]
            low_extraction_cost_element.firstChild.data = float(
                (low_extraction_cost * (1 - scale) + high_extraction_cost * scale),
            )

    return low_xml


def interpolate_cement_incelas(low_xml, high_xml, scale):
    low_region_elements = low_xml.getElementsByTagName("region")
    high_region_elements = high_xml.getElementsByTagName("region")

    for low_region_element, high_region_element in zip(
        low_region_elements,
        high_region_elements,
    ):
        low_income_elasticity_elements = low_region_element.getElementsByTagName(
            "income-elasticity",
        )
        high_income_elasticity_elements = high_region_element.getElementsByTagName(
            "income-elasticity",
        )

        for low_income_elasticity_element, high_income_elasticity_element in zip(
            low_income_elasticity_elements,
            high_income_elasticity_elements,
        ):
            low_income_elasticity = float(low_income_elasticity_element.firstChild.data)
            high_income_elasticity = float(
                high_income_elasticity_element.firstChild.data,
            )
            low_income_elasticity_element.firstChild.data = float(
                (low_income_elasticity * (1 - scale) + high_income_elasticity * scale),
            )

    return low_xml


def interpolate_industry_incelas(low_xml, high_xml, scale):
    low_region_elements = low_xml.getElementsByTagName("region")
    high_region_elements = high_xml.getElementsByTagName("region")

    for low_region_element, high_region_element in zip(
        low_region_elements,
        high_region_elements,
    ):
        low_income_elasticity_elements = low_region_element.getElementsByTagName(
            "income-elasticity",
        )
        high_income_elasticity_elements = high_region_element.getElementsByTagName(
            "income-elasticity",
        )

        for low_income_elasticity_element, high_income_elasticity_element in zip(
            low_income_elasticity_elements,
            high_income_elasticity_elements,
        ):
            low_income_elasticity = float(low_income_elasticity_element.firstChild.data)
            high_income_elasticity = float(
                high_income_elasticity_element.firstChild.data,
            )
            low_income_elasticity_element.firstChild.data = float(
                (low_income_elasticity * (1 - scale) + high_income_elasticity * scale),
            )
    return low_xml


def interpolate_wind_solar_backups(xml_lo, xml_hi, weight):
    # add intermittent-technologies
    lo_elements = xml_lo.getElementsByTagName("intermittent-technology")
    hi_elements = xml_hi.getElementsByTagName("intermittent-technology")

    for int_techs_lo, int_techs_hi in zip(lo_elements, hi_elements):
        lo_period_elements = int_techs_lo.getElementsByTagName("period")
        hi_period_elements = int_techs_hi.getElementsByTagName("period")

        # print(lo_period_elements[0])
        for low_int_period_element, high_int_period_element in zip(
            lo_period_elements,
            hi_period_elements,
        ):
            # find high capital-overnight
            hi_input_capital_element = high_int_period_element.getElementsByTagName(
                "capacity-limit-backup-calculator",
            )[0]
            hi_capital_overnight_element = (
                hi_input_capital_element.getElementsByTagName("capacity-limit")[0]
            )
            # find low capital-overnight
            lo_input_capital_element = low_int_period_element.getElementsByTagName(
                "capacity-limit-backup-calculator",
            )[0]
            lo_capital_overnight_element = (
                lo_input_capital_element.getElementsByTagName("capacity-limit")[0]
            )
            # interpolate capital overnight
            capital_overnight_interpolated = (
                weight * float(hi_capital_overnight_element.firstChild.data)
            ) + ((1 - weight) * float(lo_capital_overnight_element.firstChild.data))
            # replace capital overnight with interpolated value
            hi_capital_overnight_element.firstChild.data = (
                capital_overnight_interpolated
            )

    return xml_hi


def interpolate_wind(xml_lo, xml_hi, weightS, weightT):
    # add wind-storage
    lo_elements = xml_lo.getElementsByTagName("technology")
    hi_elements = xml_hi.getElementsByTagName("technology")

    lo_element = lo_elements[0]
    hi_element = hi_elements[0]

    lo_period_elements = lo_element.getElementsByTagName("period")
    hi_period_elements = hi_element.getElementsByTagName("period")

    for low_period_element, high_period_element in zip(
        lo_period_elements,
        hi_period_elements,
    ):
        # find high capital-overnight
        hi_input_capital_element = high_period_element.getElementsByTagName(
            "input-capital",
        )[0]
        hi_capital_overnight_element = hi_input_capital_element.getElementsByTagName(
            "capital-overnight",
        )[0]

        # find low capital-overnight
        lo_input_capital_element = low_period_element.getElementsByTagName(
            "input-capital",
        )[0]
        lo_capital_overnight_element = lo_input_capital_element.getElementsByTagName(
            "capital-overnight",
        )[0]

        # interpolate capital overnight
        capital_overnight_interpolated = (
            weightS * float(hi_capital_overnight_element.firstChild.data)
        ) + ((1 - weightS) * float(lo_capital_overnight_element.firstChild.data))

        # replace capital overnight with interpolated value
        hi_capital_overnight_element.firstChild.data = capital_overnight_interpolated
        # hi_capital_overnight_element.firstChild.data = lo_capital_overnight_element.firstChild.data
        # print(capital_overnight_interpolated)

    # add wind/wind-offshore
    lo_elements = xml_lo.getElementsByTagName("intermittent-technology")
    hi_elements = xml_hi.getElementsByTagName("intermittent-technology")

    for int_techs_lo, int_techs_hi in zip(lo_elements, hi_elements):
        lo_period_elements = int_techs_lo.getElementsByTagName("period")
        hi_period_elements = int_techs_hi.getElementsByTagName("period")

        for low_int_period_element, high_int_period_element in zip(
            lo_period_elements,
            hi_period_elements,
        ):
            # find high capital-overnight
            hi_input_capital_element = high_int_period_element.getElementsByTagName(
                "input-capital",
            )[0]
            hi_capital_overnight_element = (
                hi_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )

            # find low capital-overnight
            lo_input_capital_element = low_int_period_element.getElementsByTagName(
                "input-capital",
            )[0]
            lo_capital_overnight_element = (
                lo_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )

            # interpolate capital overnight
            capital_overnight_interpolated = (
                weightT * float(hi_capital_overnight_element.firstChild.data)
            ) + ((1 - weightT) * float(lo_capital_overnight_element.firstChild.data))

            # replace capital overnight with interpolated value
            hi_capital_overnight_element.firstChild.data = (
                capital_overnight_interpolated
            )

            # hi_capital_overnight_element.firstChild.data = lo_capital_overnight_element.firstChild.data

    return xml_hi


def interpolate_solar(xml_lo, xml_hi, weightS, weightT):
    # add CSP_storage, PV_storage
    lo_elements = xml_lo.getElementsByTagName("technology")
    hi_elements = xml_hi.getElementsByTagName("technology")

    for int_techs_lo, int_techs_hi in zip(lo_elements, hi_elements):
        lo_period_elements = int_techs_lo.getElementsByTagName("period")
        hi_period_elements = int_techs_hi.getElementsByTagName("period")

        for low_period_element, high_period_element in zip(
            lo_period_elements,
            hi_period_elements,
        ):
            # find high capital-overnight
            hi_input_capital_element = high_period_element.getElementsByTagName(
                "input-capital",
            )[0]
            hi_capital_overnight_element = (
                hi_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )

            # find low capital-overnight
            lo_input_capital_element = low_period_element.getElementsByTagName(
                "input-capital",
            )[0]
            lo_capital_overnight_element = (
                lo_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )

            # interpolate capital overnight
            capital_overnight_interpolated = (
                weightS * float(hi_capital_overnight_element.firstChild.data)
            ) + ((1 - weightS) * float(lo_capital_overnight_element.firstChild.data))

            # replace capital overnight with interpolated value
            hi_capital_overnight_element.firstChild.data = (
                capital_overnight_interpolated
            )
            # hi_capital_overnight_element.firstChild.data = lo_capital_overnight_element.firstChild.data
            # print(capital_overnight_interpolated)

    # add CSP, PV, rooftop_pv
    lo_elements = xml_lo.getElementsByTagName("intermittent-technology")
    hi_elements = xml_hi.getElementsByTagName("intermittent-technology")

    for int_techs_lo, int_techs_hi in zip(lo_elements, hi_elements):
        lo_period_elements = int_techs_lo.getElementsByTagName("period")
        hi_period_elements = int_techs_hi.getElementsByTagName("period")

        for low_int_period_element, high_int_period_element in zip(
            lo_period_elements,
            hi_period_elements,
        ):
            # find high capital-overnight
            hi_input_capital_element = high_int_period_element.getElementsByTagName(
                "input-capital",
            )[0]
            hi_capital_overnight_element = (
                hi_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )

            # find low capital-overnight
            lo_input_capital_element = low_int_period_element.getElementsByTagName(
                "input-capital",
            )[0]
            lo_capital_overnight_element = (
                lo_input_capital_element.getElementsByTagName("capital-overnight")[0]
            )

            # interpolate capital overnight
            capital_overnight_interpolated = (
                weightT * float(hi_capital_overnight_element.firstChild.data)
            ) + ((1 - weightT) * float(lo_capital_overnight_element.firstChild.data))

            # replace capital overnight with interpolated value
            hi_capital_overnight_element.firstChild.data = (
                capital_overnight_interpolated
            )


    return xml_hi


def interpolate_transpo(xml_lo, xml_hi, weight):
    lo_regions = xml_lo.getElementsByTagName("region")
    hi_regions = xml_hi.getElementsByTagName("region")
    for lo_regional_attributes, hi_regional_attributes in zip(lo_regions, hi_regions):
        # get fuelprefElasticities
        fuelprefElasticities_lo = lo_regional_attributes.getElementsByTagName(
            "fuelprefElasticity",
        )
        fuelprefElasticities_hi = hi_regional_attributes.getElementsByTagName(
            "fuelprefElasticity",
        )
        for lo_fuelprefs, hi_fuelprefs in zip(
            fuelprefElasticities_lo,
            fuelprefElasticities_hi,
        ):
            fuelprefs_interpolated = (weight * float(hi_fuelprefs.firstChild.data)) + (
                (1 - weight) * float(lo_fuelprefs.firstChild.data)
            )
            hi_fuelprefs.firstChild.data = fuelprefs_interpolated

        # get speeds
        lo_speeds = lo_regional_attributes.getElementsByTagName("speed")
        hi_speeds = hi_regional_attributes.getElementsByTagName("speed")
        for lo_speed, hi_speed in zip(lo_speeds, hi_speeds):
            speed_interpolated = (weight * float(hi_speed.firstChild.data)) + (
                (1 - weight) * float(lo_speed.firstChild.data)
            )

            hi_speed.firstChild.data = str(int(speed_interpolated))

        # get income-elasticities
        energy_final_demands_lo = lo_regional_attributes.getElementsByTagName(
            "energy-final-demand",
        )
        energy_final_demands_hi = hi_regional_attributes.getElementsByTagName(
            "energy-final-demand",
        )
        for final_demands_lo, final_demands_hi in zip(
            energy_final_demands_lo,
            energy_final_demands_hi,
        ):
            lo_income_elasticities = final_demands_lo.getElementsByTagName(
                "income-elasticity",
            )
            hi_income_elasticities = final_demands_hi.getElementsByTagName(
                "income-elasticity",
            )
            for lo_income_elasticity, hi_income_elasticity in zip(
                lo_income_elasticities,
                hi_income_elasticities,
            ):
                elasticity_interpolated = (
                    weight * float(hi_income_elasticity.firstChild.data)
                ) + ((1 - weight) * float(lo_income_elasticity.firstChild.data))
                hi_income_elasticity.firstChild.data = elasticity_interpolated
                # hi_income_elasticity.firstChild.data = lo_income_elasticity.firstChild.data

        # get time-value-multipliers
        lo_TVMs = lo_regional_attributes.getElementsByTagName("time-value-multiplier")
        hi_TVMs = hi_regional_attributes.getElementsByTagName("time-value-multiplier")
        for lo_TVM, hi_TVM in zip(lo_TVMs, hi_TVMs):
            TVM_interpolated = (weight * float(hi_TVM.firstChild.data)) + (
                (1 - weight) * float(lo_TVM.firstChild.data)
            )
            hi_TVM.firstChild.data = TVM_interpolated
            # hi_TVM.firstChild.data = lo_TVM.firstChild.data

        # get addTimeValues
        lo_aTVs = lo_regional_attributes.getElementsByTagName("addTimeValue")
        hi_aTVs = hi_regional_attributes.getElementsByTagName("addTimeValue")
        for lo_aTV, hi_aTV in zip(lo_aTVs, hi_aTVs):
            aTV_interpolated = (weight * float(hi_aTV.firstChild.data)) + (
                (1 - weight) * float(lo_aTV.firstChild.data)
            )
            hi_aTV.firstChild.data = str(int(aTV_interpolated))
            # hi_aTV.firstChild.data = lo_aTV.firstChild.data

        # get coefficients and fuelprefElasticities
        lo_stub_techs = lo_regional_attributes.getElementsByTagName("stub-technology")
        hi_stub_techs = hi_regional_attributes.getElementsByTagName("stub-technology")
        for lo_elements, hi_elements in zip(lo_stub_techs, hi_stub_techs):
            # get load factors
            load_factors_lo = lo_elements.getElementsByTagName("loadFactor")
            load_factors_hi = hi_elements.getElementsByTagName("loadFactor")
            for load_lo, load_hi in zip(load_factors_lo, load_factors_hi):
                loadFactor_interpolated = (weight * float(load_hi.firstChild.data)) + (
                    (1 - weight) * float(load_lo.firstChild.data)
                )
                load_hi.firstChild.data = loadFactor_interpolated

            # get input costs
            input_cost_lo = lo_elements.getElementsByTagName("input-cost")
            input_cost_hi = hi_elements.getElementsByTagName("input-cost")
            for cost_lo, cost_hi in zip(input_cost_lo, input_cost_hi):
                input_cost_interpolated = (weight * float(cost_hi.firstChild.data)) + (
                    (1 - weight) * float(cost_lo.firstChild.data)
                )
                cost_hi.firstChild.data = input_cost_interpolated

            # get coefficients
            lo_coefficients = lo_elements.getElementsByTagName("coefficient")
            hi_coefficients = hi_elements.getElementsByTagName("coefficient")
            for lo_coefficient, hi_coefficient in zip(lo_coefficients, hi_coefficients):
                coefficient_interpolated = (
                    weight * float(hi_coefficient.firstChild.data)
                ) + ((1 - weight) * float(lo_coefficient.firstChild.data))
                hi_coefficient.firstChild.data = coefficient_interpolated


    return xml_hi


def open_xml(path: str):
    root_path = Path(config.paths.repo) / "ml_gcam" / "core" / "inputs"
    return minidom.parse(str(root_path / path))


def create_intperolated_xmls(
    interpolated_scales: List[float] = [0.5] * 12,
    paths_object: XMLPaths = PATHS,
):
    """Ff emiss bio nuc ccs back windS windT solarS solarT."""
    (
        back,
        _,
        ccs,
        elec,
        _,
        eng,
        ff,
        nuc,
        solarS,
        solarT,
        windS,
        windT,
    ) = interpolated_scales

    resources_xml = interpolate_resources(
        open_xml(paths_object.RECSSP1),
        open_xml(paths_object.RECSSP5),
        ff,
    )
    nuclear_xml = interpolate_nuclear(
        open_xml(paths_object.NUCL),
        open_xml(paths_object.NUCH),
        nuc,
    )
    ccs_suply_xml = intperpolate_ccs_supply(
        open_xml(paths_object.CCSLO),
        open_xml(paths_object.CCSHI),
        ccs,
    )
    wind_solar_backups_xml = interpolate_wind_solar_backups(
        open_xml(paths_object.WSBL),
        open_xml(paths_object.WSBH),
        back,
    )

    wind_xml = interpolate_wind(
        open_xml(paths_object.WINDL),
        open_xml(paths_object.WINDA),
        windS,
        windT,
    )
    solar_xml = interpolate_solar(
        open_xml(paths_object.SOLL),
        open_xml(paths_object.SOLA),
        solarS,
        solarT,
    )

    if elec:
        industry_incelas_xml = interpolate_industry_incelas(
            open_xml(paths_object.INDSSP1E),
            open_xml(paths_object.INDSSP5E),
            eng,
        )
        building_xml = interpolate_building(
            open_xml(paths_object.BSSP1E),
            open_xml(paths_object.BSSP5E),
            eng,
        )
        transportation_xml = interpolate_transpo(
            open_xml(paths_object.TRANSSP1E),
            open_xml(paths_object.TRANSSP5E),
            eng,
        )
    else:
        industry_incelas_xml = interpolate_industry_incelas(
            open_xml(paths_object.INDSSP1),
            open_xml(paths_object.INDSSP5),
            eng,
        )
        building_xml = interpolate_building(
            open_xml(paths_object.BSSP1),
            open_xml(paths_object.BSSP5),
            eng,
        )
        transportation_xml = interpolate_transpo(
            open_xml(paths_object.TRANSSP1),
            open_xml(paths_object.TRANSSP5),
            eng,
        )

    cement_incelas_xml = interpolate_cement_incelas(
        open_xml(paths_object.CEMSSP1),
        open_xml(paths_object.CEMSSP5),
        eng,
    )
    socioeconomics_xml = interpolate_socioeconomics(
        open_xml(paths_object.SOCSSP1),
        open_xml(paths_object.SOCSSP5),
        eng,
    )

    return (
        resources_xml,
        nuclear_xml,
        ccs_suply_xml,
        wind_solar_backups_xml,
        wind_xml,
        solar_xml,
        industry_incelas_xml,
        building_xml,
        transportation_xml,
        cement_incelas_xml,
        socioeconomics_xml,
    )

    # For elec that is hi use that xml corrosponding to that.

    # Weird, emiss, bio,


def create_xml_packages(interpolated_scales):
    final_xmls = {}
    xmls = create_intperolated_xmls(interpolated_scales)
    (
        resources_xml,
        nuclear_xml,
        ccs_suply_xml,
        wind_solar_backups_xml,
        wind_xml,
        solar_xml,
        industry_incelas_xml,
        building_xml,
        transportation_xml,
        cement_incelas_xml,
        socioeconomics_xml,
    ) = xmls

    final_xmls["resources.xml"] = resources_xml
    final_xmls["2025_target_finder.xml"] = open_xml(PATHS.TF2025)
    final_xmls["spa14_tax.xml"] = open_xml(PATHS.SPA14)
    if interpolated_scales[1] == 0.0:
        final_xmls["bio_constraint_100_hi.xml"] = open_xml(PATHS.BIO)
    final_xmls["nuclear.xml"] = nuclear_xml
    final_xmls["ccs_supply.xml"] = ccs_suply_xml
    final_xmls["wind_solar_backups.xml"] = wind_solar_backups_xml
    final_xmls["wind.xml"] = wind_xml
    final_xmls["solar.xml"] = solar_xml
    final_xmls["industry_incelas.xml"] = industry_incelas_xml
    final_xmls["building.xml"] = building_xml
    final_xmls["transportation_UCD.xml"] = transportation_xml
    final_xmls["cement_incelas.xml"] = cement_incelas_xml
    final_xmls["socioeconomics.xml"] = socioeconomics_xml

    return final_xmls


def create_input_directory(name, bits):
    # interpolated_scales = generate_interpolated_bitmap()
    final_xmls = create_xml_packages(bits)
    # ff, emiss, bio, nuc, ccs, back, windS, windT, solarS, solarT, eng, elec = bits
    (
        back,
        bio,
        ccs,
        elec,
        emiss,
        energy,
        ff,
        nuc,
        solarS,
        solarT,
        windS,
        windT,
    ) = bits

    dir_name = f"ff-{ff}_emiss-{emiss}_bio-{bio}_nuc-{nuc}_ccs-{ccs}_back-{back}_windS-{windS}_windT-{windT}_solarS-{solarS}_solarT-{solarT}_energy-{energy}_elec-{elec}"
    parent = Path(config.paths.interpolation) / name / "input" / dir_name
    if not parent.exists():
        parent.mkdir(exist_ok=True, parents=True)
    for file_name, xml in final_xmls.items():
        with open(parent / file_name, "w") as f:
            xml.writexml(f, indent="  ", addindent="  ")
    return parent, bits


def make_inputs(num_to_create, name):
    """Create paths containing interpolated inputs."""
    results = []
    samples = sample_bits(num_to_create, name)
    for i in (bar := tqdm(range(len(samples)))):
        bar.set_description(f"{samples[i]}")
        result = {}
        result["path"], result["bits"] = create_input_directory(name, samples[i])
        results.append(result)
    df = pd.DataFrame([x["bits"] for x in results], columns=config.data.input_keys)
    df["path"] = [x["path"] for x in results]
    save_to = Path(config.paths.data) / "meta" / f"{name}_scenarios.csv"
    df.to_csv(save_to, sep="|", index=False)


def make_configs(name):
    """Create configs from set of interpolated inputs."""
    parent = Path(config.paths.interpolation) / name
    columns = config.data.input_keys
    meta = pd.read_csv(
        Path(config.paths.data) / "meta" / f"{name}_scenarios.csv",
        sep="|",
    )

    for _, row in (bar := tqdm(meta.iterrows())):
        bits = row[columns].values
        if bits[1] == 0.0:
            if bits[4] == 0.0:
                template = Path(config.paths.repo) / "ml_gcam" / "core" /  "config_templates" / "gcam_config_template_bio-lo_emiss-lo.xml"
            else:
                template = Path(config.paths.repo) / "ml_gcam" / "core" /  "config_templates" / "gcam_config_template_bio-lo_emiss-hi.xml"
        else:
            if bits[4] == 0.0:
                template = Path(config.paths.repo) / "ml_gcam" / "core" /  "config_templates" / "gcam_config_template_bio-hi_emiss-lo.xml"
            else:
                template = Path(config.paths.repo) / "ml_gcam" / "core" /  "config_templates" / "gcam_config_template_bio-hi_emiss-hi.xml"
        d = Path(row.path)
        scenario_input_path = parent / "input" / f"{d.name}"
        bar.set_description(f"{row.name}: {d.name}")

        doc = etree.parse(template)
        component = doc.xpath("//ScenarioComponents[2]")[0]

        # change the output database path
        doc.xpath("//Files/Value[@name='xmldb-location']")[
            0
        ].text = f"{parent / 'output'}/"

        # change the scneario name
        scenario_name = f"scenario_{row.name}"
        doc.xpath("//Strings/Value[@name='scenarioName']")[0].text = scenario_name

        # update the path to config files
        mapping = {
            "fossil_costs": "resources.xml",
            "nuclear": "nuclear.xml",
            "ccs": "ccs_supply.xml",
            "backups": "wind_solar_backups.xml",
            "wind": "wind.xml",
            "solar": "solar.xml",
            "industry": "industry_incelas.xml",
            "buildings": "building.xml",
            "transport": "transportation_UCD.xml",
            "cement": "cement_incelas.xml",
            "pop_gdp": "socioeconomics.xml",
        }

        for name, xml in mapping.items():
            to_change = component.xpath(f"./Value[@name='{name}']")[0]
            change_to = scenario_input_path / xml
            to_change.text = str(change_to)

        # save modified config template
        save_to = parent / "config" / (scenario_name + ".xml")
        if not save_to.parent.exists():
            save_to.parent.mkdir(exist_ok=True)
        save_to.touch(exist_ok=True)
        with open(save_to, "wb") as f:
            f.write(etree.tostring(doc))
