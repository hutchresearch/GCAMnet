from pathlib import Path
import numpy as np
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


def sigma_normalization_factor(inputs_std, output_std, epsilon: float = 1e-12):
    """Compute sigma-normalization factor with stable handling for tiny output std.

    DGSM normalization is sigma_x / sigma_y. For flat outputs (sigma_y ~= 0),
    we clamp the denominator to avoid inf/NaN propagation.
    """
    denom = output_std if abs(float(output_std)) > epsilon else epsilon
    return inputs_std / denom


def dgsm_sensitivity_compare(
    targets_path,
    train_source,
    checkpoint_path: Path = None,
    save_path: Path = None,
    dgsm: str = None,
    strategy = 'z_score',
    aggregation: str = 'none',
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

    # Pre-compute z-score statistics for cross-quantity aggregation (region/year modes)
    if aggregation in ('region', 'year'):
        q_mean = gcam_y.mean()          # per-quantity mean across all (region, year, sample)
        q_std  = gcam_y.std()           # per-quantity std
        valid_q = q_std[q_std >= 1e-8].index.tolist()  # drop zero-variance quantities

    if aggregation == 'none':
        agg_keys = config.data.output_keys
    elif aggregation == 'quantity':
        agg_keys = config.data.output_keys
    elif aggregation == 'region':
        agg_keys = config.data.region_keys
    elif aggregation == 'year':
        agg_keys = [str(y) for y in config.data.year_keys]

    sp = ProblemSpec(
        {
            "num_vars": D,
            "names": input_keys,
            "bounds": [[0, 1]] * D,
            "outputs": agg_keys,
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

                inputs_std = sample_subset.std().to_numpy().flatten()

                # --- Aggregation path (quantity / region / year) ---
                if aggregation != 'none':
                    # Accumulate aggregated outputs: shape (len(take), len(agg_keys))
                    y_true_agg = np.zeros((len(take), len(agg_keys)))
                    y_pred_agg = np.zeros((len(take), len(agg_keys)))

                    for region, year in pbar:
                        y_true = gcam_y.xs((region, str(year))).iloc[take]
                        y_pred = emulator_y.xs((region, str(year))).iloc[take]

                        if aggregation == 'quantity':
                            # Sum raw outputs across (region, year) for each quantity
                            for qi, q in enumerate(agg_keys):
                                y_true_agg[:, qi] += y_true[q].values
                                y_pred_agg[:, qi] += y_pred[q].values

                        elif aggregation == 'region':
                            # Z-score each quantity, then accumulate into the region slice
                            ri = config.data.region_keys.index(region)
                            for q in valid_q:
                                y_true_agg[:, ri] += (y_true[q].values - q_mean[q]) / q_std[q]
                                y_pred_agg[:, ri] += (y_pred[q].values - q_mean[q]) / q_std[q]

                        elif aggregation == 'year':
                            yi = [str(y) for y in config.data.year_keys].index(str(year))
                            for q in valid_q:
                                y_true_agg[:, yi] += (y_true[q].values - q_mean[q]) / q_std[q]
                                y_pred_agg[:, yi] += (y_pred[q].values - q_mean[q]) / q_std[q]

                    # Convert sums to averages so vi values are comparable to non-aggregated
                    if aggregation == 'quantity':
                        n_terms = len(config.data.region_keys) * len(config.data.year_keys)
                    elif aggregation == 'region':
                        n_terms = len(config.data.year_keys) * len(valid_q)
                    elif aggregation == 'year':
                        n_terms = len(config.data.region_keys) * len(valid_q)
                    y_true_agg /= n_terms
                    y_pred_agg /= n_terms

                    # Run DGSM on the aggregated outputs
                    core_agg_std = np.std(y_true_agg, axis=0)  # std per aggregated key

                    core, emulator = [], []
                    for label, collect, y_agg in [
                        ("core", core, y_true_agg),
                        ("emulator", emulator, y_pred_agg),
                    ]:
                        sp = sp.set_results(y_agg)
                        sp.analyze_dgsm()
                        for ki, key in enumerate(agg_keys):
                            s1 = sp.analysis[key][dgsm]
                            if dgsm == "vi":
                                sigma_norm = sigma_normalization_factor(
                                    inputs_std,
                                    core_agg_std[ki],
                                )
                                s1 = s1 * sigma_norm

                            s1 = dict(zip(input_keys, s1))

                            row = {
                                "train_source": str(train_source),
                                "dev_source": "interp_dgsm",
                                aggregation: key,
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
                    continue
                # --- End aggregation path; fall through to per-(region, year) for 'none' ---

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
                            s1 = sp.analysis[feature][dgsm]
                            if dgsm == "vi":
                                sigma_norm = sigma_normalization_factor(
                                    inputs_std,
                                    core_std[feature],
                                )
                                s1 = s1 * sigma_norm
                            conf_key = "dgsm_conf" if dgsm == "dgsm" else "vi_std"
                            conf = sp.analysis[feature][conf_key]

                            s1 = dict(zip(input_keys, s1))

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
