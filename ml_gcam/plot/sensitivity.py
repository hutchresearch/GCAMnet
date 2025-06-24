from pathlib import Path
import polars as pl
import numpy as np 
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

from ..data import experiment_name_to_paper_label
from .. import config, logger


def sensitivity_one_to_one(world_paths: list, cumulative: bool, kde:bool, train_source: str):
    '''entry point for cli.plot:sensitivity'''
    if kde:
        fig = kde_one_to_one(world_paths, train_source)
    else:
        fig = histogram(world_paths, train_source)
    return fig

    
def histogram(worlds, train_source):
    '''plot the histgoram of core vs. emulator'''
    collect = []
    for file in worlds:
        df = pl.scan_csv(file, separator='|').collect()
        collect.append(df)
    df = pl.concat(collect)
    
    to_drop = ["bio", "elec", "emiss"]
    input_keys = [i for i in config.data.input_keys if i not in to_drop]

    emulator_df = df.filter(pl.col("source")=="emulator")[input_keys]
    core_df = df.filter(pl.col("source")=="core")[input_keys]

    fig = plt.figure(figsize=(20, 20))

    bins = np.arange(-2, 200, 5)
    plt.hist(emulator_df.to_numpy().flatten(), alpha=0.5, bins=bins, label="emulator")
    plt.hist(core_df.to_numpy().flatten(), alpha=0.5, bins=bins, label="core")

    plt.xlabel('DGSM Value')
    plt.ylabel('Count')
    plt.title(f"Core vs {experiment_name_to_paper_label(train_source).capitalize()} Emulator Sensitivities", fontsize=16)
    plt.legend()

    return fig


def kde_one_to_one(worlds, train_source):
    '''plot kde one-to-one plots for core vs. emulator'''
    collect = []
    for file in worlds:
        df = pl.scan_csv(file, separator='|').collect()
        collect.append(df)
    df = pl.concat(collect)
    
    to_drop = ["bio", "elec", "emiss"]
    input_keys = [i for i in config.data.input_keys if i not in to_drop]

    emulator_df = df.filter(pl.col("source")=="emulator")[input_keys]
    core_df = df.filter(pl.col("source")=="core")[input_keys]

    fig = make_one_to_one(
        true=core_df,
        predictions=emulator_df,
        train_source=train_source,
        kde_bandwidth=10,
    )

    return fig


def sensitivity_heatmaps(sensitivity_path: Path):
    '''plot sensitivity heatmap for all sampling strategies'''

    fig = plt.figure(figsize=(18,12))

    spec = gridspec.GridSpec(
        ncols=5, nrows=3, width_ratios=[3, 3, 3, 3, 0.2], wspace=0.25, hspace=0.25
    )

    plt.rc("axes", titlesize=9)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=7)
    plt.tight_layout()
    
    # plt.suptitle("Derivative-Based Sensitivity Measure by Sampling Strategy", fontsize=18)

    source_data = {}
    for source in sensitivity_path.iterdir():
        collect = []
        for csv in sorted(source.iterdir()):
            df = pl.scan_csv(csv, separator="|").collect()
            collect.append(df)
        source_data[source.stem] = pl.concat(collect)

    to_drop = ["bio", "elec", "emiss"]
    input_keys = [i for i in config.data.input_keys if i not in to_drop]
    index = ['year', 'quantity', 'region']
    metric_keys = {}
    metric_keys["year"] = config.data.year_keys
    metric_keys["quantity"] = config.data.output_keys
    metric_keys["region"] = config.data.region_keys

    for row, metric in enumerate(index):
        for col, source_key in enumerate(["core", "binary", "interp_hypercube", "mixed"]):

            ax = fig.add_subplot(spec[row, col])
            if row == 0:
                ax.set_title(f"{experiment_name_to_paper_label(source_key).capitalize()}", fontsize=16)
            if col == 0:
                ax.set_xlabel(metric.capitalize(), fontsize=16)
        
            if col == 0:
                yticklabels = metric_keys[metric]
            else:
                yticklabels = False
            if row == 2:
                xticklabels = input_keys
            else:
                xticklabels = False
            
            data = source_data["binary"].filter(pl.col("source") == "core") if source_key == "core" else source_data[source_key].filter(pl.col("source") == "emulator")
            grouped = data.groupby(metric).median().sort(metric)[input_keys]

            map_val = sns.heatmap(
                grouped,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot=False,
                vmin=0,
                vmax=5,
                cmap=sns.color_palette("Blues", as_cmap=True),
                cbar=False,
                ax=ax
            )

            map_val.set_yticklabels(map_val.get_yticklabels(), rotation=0)
            map_val.set_xlabel("")
            map_val.set_ylabel("")

            if source_key != "core":
                core_data = source_data[source_key].filter(pl.col("source") == "core")
                core_grouped = core_data.groupby(metric).median().sort(metric)[input_keys]
                map_r2 = r2_score(core_grouped.to_numpy().flatten(), grouped.to_numpy().flatten())


                # label = "R2: " + str('%.3f'%map_r2)
                # if metric == "region":
                #     map_r2_overall = r2_score(
                #         core_data[input_keys].to_numpy().flatten(), data[input_keys].to_numpy().flatten()
                #     )
                #     label = "R2: " + str('%.3f'%map_r2) + " | R2 Overall: " + str('%.3f'%map_r2_overall)

                # ax.text(
                #         0.95,
                #         -0.2,
                #         label,
                #         fontsize=8,
                #         ha="right",
                #         va="bottom",
                #         transform=ax.transAxes,
                #     )

    # Add a color bar in the fourth column of the grid, spanning all rows
    cbar_ax = fig.add_subplot(spec[:, 4])

    # Use one of the heatmaps (e.g., hm1) to create the color bar
    map_val.figure.colorbar(map_val.collections[0], cax=cbar_ax)
    return fig


def sensitivity_sample_size(dir_path: Path):
    '''plot sensitivty r2 score per metric vs. sample size'''
    samples = [world for world in dir_path.iterdir()]
    
    sample_collect = []
    for sample in samples:
        collect = []
        for file in sample.iterdir():
            df = pl.scan_csv(file, separator='|').collect()
            collect.append(df)
        df = pl.concat(collect)
        
        to_drop = ["bio", "elec", "emiss"]
        input_keys = [i for i in config.data.input_keys if i not in to_drop]

        emulator_df = df.filter(pl.col("source")=="emulator")
        core_df = df.filter(pl.col("source")=="core")

        overall_r2 = r2_score(core_df[input_keys].to_numpy().flatten(), emulator_df[input_keys].to_numpy().flatten())

        metric_collect = {}
        metric_collect["train"] = "mixed"
        metric_collect["test"] = "dgsm"
        metric_collect["sample_size"] = int(sample.name.split("_")[0])

        for metric in ["region", "year", "quantity"]:
            grouped_core = core_df.groupby(metric).median().sort(metric)[input_keys]
            grouped_emulator = emulator_df.groupby(metric).median().sort(metric)[input_keys]
            r2 = r2_score(grouped_core.to_numpy().flatten(), grouped_emulator.to_numpy().flatten())
            metric_collect[metric] = r2
        metric_collect["overall"] = overall_r2
        sample_collect.append(pl.DataFrame(metric_collect))
    table = pl.concat(sample_collect)

    fig = plt.figure(figsize=(4,4))
    
    for metric in ["region", "quantity", "year", "overall"]:
        sns.lineplot(table, x="sample_size", y=metric, label=metric)
    plt.xlabel("Training Samples")
    plt.ylabel("$R^2$ Score")
    plt.xlim([0,3300])
    plt.ylim([0,1.01])
    plt.title("$R^2$ Score per Metric vs. Number of Training Samples")
    plt.legend()
    plt.tight_layout()
    return fig
    

def remove_outliers(core_df, em_df):

    outlier_indices = []
    core_df = core_df.to_pandas()
    em_df = em_df.to_pandas()
    core_df.reset_index(drop=True, inplace=True)
    em_df.reset_index(drop=True, inplace=True)

    to_drop = ["bio", "elec", "emiss"]
    input_keys = [i for i in config.data.input_keys if i not in to_drop]

    threshold = np.quantile(np.vstack([core_df.values, em_df.values]), 0.99)

    # find core outliers
    for column in input_keys:
        quant = threshold

        outliers = core_df[core_df[column] > quant]
        outlier_indices = outlier_indices + outliers.index.tolist()

        nans = core_df[core_df.isna().any(axis=1)]
        outlier_indices = outlier_indices + nans.index.tolist()

        infs = core_df[np.isinf(core_df).any(axis=1)]
        outlier_indices = outlier_indices + infs.index.tolist()   

    #find emulator outliers
    for column in input_keys:
        quant = threshold

        outliers = em_df[em_df[column] > quant]
        outlier_indices= outlier_indices + outliers.index.tolist()

        nans = em_df[em_df.isna().any(axis=1)]
        outlier_indices = outlier_indices + nans.index.tolist()

        infs = em_df[np.isinf(em_df).any(axis=1)]
        outlier_indices = outlier_indices + infs.index.tolist()
        
    core_new = core_df.drop(set(outlier_indices))
    em_new = em_df.drop(set(outlier_indices))

    return pl.from_pandas(core_new), pl.from_pandas(em_new)


def make_one_to_one(
        true,
        predictions,
        train_source,
        kde_bandwidth=0.01,
        grid_size=2 ** 8,
        y_range=(0,5),
        x_range=(0,5),
    ):
    np_true = true.to_numpy().flatten()
    np_predictions = predictions.to_numpy().flatten()
    
    true, predictions = remove_outliers(true, predictions)
    np_true = true.to_numpy().flatten()
    np_predictions = predictions.to_numpy().flatten()

    true_pred = np.vstack([np_true, np_predictions]).T
    pred_probs = calc_KDE(true_pred, kde_bandwidth, grid_size)
        
    pred_probs = np.clip(pred_probs, a_min=0, a_max =None)

    fig, ax = plt.subplots()

    plt.plot([-200000, 9422070600 ], [-200000, 9422070600 ], c="k", linewidth=1)

    one_to_one = ax.scatter(
        true,
        predictions,
        c=pred_probs,
        cmap="plasma",
        s=1,
        linewidths=0
    )

    cbar = fig.colorbar(one_to_one, ax=ax)
    cbar.set_label("density")

    max_value = max(true_pred.flatten())
    max_value = 60
    x_range = (x_range[0], max_value)
    y_range = (y_range[0], max_value)

    plt.ylim(*y_range)
    plt.xlim(*x_range)

    plt.title(f"Core vs {experiment_name_to_paper_label(train_source).capitalize()} Emulator Sensitivities", fontsize=16)
    plt.xlabel("Core", fontsize=15)
    plt.ylabel("Emulator", fontsize=15)

    return fig

def calc_KDE(labels, kde_bandwidth: float, grid_size):
    """
        This will calculate the pdf from a KDE of labels (Size([num_datapoints, num_labels])).
        Note: grid_size will significantly slow things down if it is too large. 2**8 works for a
        Nx2 array, but 2**3 works for an Nx3 array. Anything more may not be worth doing.
    """
    from KDEpy import FFTKDE
    import scipy

    assert(len(labels.shape) <= 2), "Incorrect shape for labels tensor."
    
    #this will ensure gridsize is of size of all the data
    min_val = np.min(labels)
    max_val = np.max(labels)
    
    # labels = np.clip(labels, a_min=None, a_max = 1e9)

    if kde_bandwidth == None:
        kde_bandwidth=1

    labels_linear_grid, grid_probs = FFTKDE(kernel="gaussian", bw=kde_bandwidth).fit(labels, weights=None).evaluate(grid_size)

    if labels.shape[-1] == 1 or len(labels.shape) == 1:
        # For 1D arrays
        interp = scipy.interpolate.interp1d(labels_linear_grid, grid_probs)
        labels_probs = interp(labels)
    else:
        # For ND arrays
        interp = scipy.interpolate.LinearNDInterpolator(labels_linear_grid, grid_probs)
        labels_split = [label_type.squeeze() for label_type in np.split(labels, labels.shape[1], axis=1)]
        labels_probs = interp(*labels_split)

    return labels_probs
