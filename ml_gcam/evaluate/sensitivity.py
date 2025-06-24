from pathlib import Path
import pandas as pd
import polars as pl
from tqdm import tqdm
from itertools import product

from SALib import ProblemSpec
from SALib.analyze import sobol
from SALib.sample import saltelli


from .. import config, logger
from ..inference import Inference
from ..data import GcamDataset, Source, Split, load_targets
from ..data.normalization import Normalization


def dgsm_sensitivity_compare(
    targets_path, 
    train_source, 
    checkpoint_path: Path = None, 
    save_path: Path = None, 
    dgsm: str = None, 
    strategy = 'z_score',
):
    """generate s1 values from pretrained emulator weights"""
    
    denorm_dataset = GcamDataset.from_targets(
                save_path=targets_path,
                experiment=train_source,
                split=Split.TRAIN,
            )
    normalization = Normalization(outputs=denorm_dataset.outputs, strategy=strategy)

    gcam_core = GcamDataset.from_targets(
            save_path=targets_path,
            experiment=Source.DGSM,
            split=Split.DEV,
        )
    gcam_core.with_normalization(normalization)

    inference = Inference.from_checkpoint(checkpoint_path).eval_with(gcam_core).denormalize_with(normalization)

    #y_true_df and y_pred_df are dfs with the denormalized values
    gcam_y = inference.y_true_df.to_pandas().set_index(['region', 'year'])
    emulator_y = inference.y_pred_df.to_pandas().set_index(['region', 'year'])
    inputs = inference.x_df.to_pandas()
    
    # Define the problem
    to_drop = ["bio", "elec", "emiss"]
    input_keys = [i for i in config.data.input_keys if i not in to_drop]

    D = len(input_keys)

    sp = ProblemSpec(
        {
            "num_vars": D,
            "names": input_keys,
            "bounds": [[0, 1]] * D,
            "outputs": config.data.output_keys,
        }
    )
    for bio in [0, 1]:
        for emiss in [0, 1]:
            for elec in [0, 1]:
                triplet_indices = inputs[(inputs["elec"]==float(elec)) & (inputs["emiss"]==float(emiss)) & (inputs["bio"]==float(bio))].reset_index(drop=True).index
                take = triplet_indices[0: -(len(triplet_indices) % (D + 1))] if (len(triplet_indices) % (D + 1)) != 0 else triplet_indices
                logger.debug(f"Collected {len(take)} Samples for [{bio}, {elec}, {emiss}]")

                sample_subset = inference.x_df.filter(
                        (pl.col('bio') == bio),
                        (pl.col('elec') == elec),
                        (pl.col('emiss') == emiss)).drop(to_drop)
                sp = sp.set_samples(sample_subset.to_pandas().iloc[take].to_numpy())

                pbar = tqdm(
                    list(product(config.data.region_keys, config.data.year_keys)),
                    desc="dgsm calcs",
                    leave=False,
                    ncols=100,
                )
                
                inputs_std = sample_subset.std()

                core, emulator = [], []
                for region, year in pbar:
                    y_true = gcam_y.xs((region, str(year))).iloc[take]
                    y_pred = emulator_y.xs((region, str(year))).iloc[take]

                    core_std = y_true.std()

                    for label, collect, y in [
                        ("core", core, y_true),
                        ("emulator", emulator, y_pred),
                    ]:
                        sp = sp.set_results(y.values)
                        sp.analyze_dgsm()
                        for feature in config.data.output_keys:
                            sigma_norm = inputs_std / core_std[feature]
                            
                            s1 = sp.analysis[feature][dgsm]
                            s1 = s1 * sigma_norm
                            conf = "dgsm_conf" if dgsm == "dgsm" else "vi_std"
                            conf = sp.analysis[feature][conf]
                            
                            s1 = dict(zip(input_keys, sp.analysis[feature][dgsm]))
                                
                            row = {
                                "train_source": str(train_source),
                                "dev_source": "interp_dgsm",
                                "region": region,
                                "year": year,
                                "quantity": feature,
                                "source": label,
                            }
                            row |= s1
                            collect.append(row)
                filename = f'/{dgsm}_train:{train_source}_elec{elec}_emiss{emiss}_bio{bio}.csv'
                save_to = save_path + filename

                results = []
                for rows in [core, emulator]:
                    df = pd.DataFrame(rows)
                    results.append(df)
                out = pd.concat(results)
                out.to_csv(save_to, sep="|", index=False)
                
                logger.info(f"saved: {save_to}")
