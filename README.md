# GCAM Machine Learning Emulator

This repo contains an implemention of a flexible, machine learning-based Global Change Analysis Model (GCAM) emulator.

[Project Home](https://github.com/hutchresearch/ml_climate_gcam22)

## Project Scope

- Develop and train neural-network on data derived from experiments defined in [Woodward (2023)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022EF003442) (commonly references as "exp 1 jr.").
- Compare and contrast the performance of the emulator and the 'core' GCAM model performance.
- Compare and contract the sensitivity of the emulator and the 'core' GCAM model performance.
- Experiment with the sensitivity of emulator performance as a function of the amount of training data.

## People

**Researchers**

- Brian Hutchinson [email](mailto:Brian.Hutchinson@wwu.edu)
- Abigail Snyder [email](mailto:abigail.snyder@pnnl.gov)
- Claudia Tebaldi [email](mailto:claudia.tebaldi@pnnl.gov)

**Students**

- Andrew Holmes
- Hidemi Mitani Shen
- Matt Jensen
- Sarah Coffland
- Logan Sizemore
- Seth Bassetti
- Brenn Nieva

**Water Scarcity Modeling**

- Jonathan Lamontagne [email](mailto:jonathan.lamontagne@tufts.edu)
- Flannery Dolan [email](mailto:flannery.dolan@tufts.edu)
- Dawn Woodard [email](mailto:dawn.woodard@pnnl.gov)

## GCAM Background

- [GCAM core repo](https://github.com/JGCRI/gcam-core)
- [GCAM build process](http://jgcri.github.io/gcam-doc/gcam-build.html)
- [GCAMDATA build process](https://jgcri.github.io/gcamdata/articles/getting-started/getting-started.html)
- [GCAM background paper](https://gmd.copernicus.org/articles/12/677/2019/gmd-12-677-2019.pdf)
- [GCAM post processing](https://github.com/JGCRI/gcamreader/)
- [GACM video - intro](https://www.youtube.com/watch?v=xRF9lFwtMr0)
- [GACM video - tutorial](https://www.youtube.com/watch?v=S7vAShH-dbs)

## Setup

### UV Environment

To set up your environment using UV, follow these steps:

- First, ensure you have [UV](https://docs.astral.sh/uv/) installed on your system.

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- Clone the repository to your local machine:

```
git clone https://github.com/JGCRI/gcam-emulator.git
cd gcam-emulator
```

- Create a UV environment using the pyproject.toml file provided in the repository:

```
uv sync

[optional]
uv sync --no-cache
```

- Use the newly created environment:

```
# ideal way
uv run -m ml_gcam --help 

or 

# optional way
source .venv/bin/activate
python3 -m ml_gcam --help
```

### Conda Environment

To set up your environment using Conda, follow these steps:

- First, ensure you have [Conda](https://docs.anaconda.com/free/miniconda/) installed on your system.
- Clone the repository to your local machine:

```
git clone https://github.com/JGCRI/gcam-emulator.git
cd gcam-emulator
```

- Create a Conda environment using the environment.yml file provided in the repository:

```
conda env create -f environment.yml
# or with the Makefile:
# make conda_update
```

- Activate the newly created environment:

```
conda activate ml_climate_gcam22
```

### Virtualenv Environment

To set up your environment using venv, follow these steps:

- Clone the repository to your local machine:

```
git clone https://github.com/JGCRI/gcam-emulator.git
cd gcam-emulator
```

- Create a virtualenv environment using `venv` and the requirements.txt file provided in the repository:

```
# install virtualenv
python3 -m venv .venv

# load the python environment
source .venv/bin/activate
pip install -r requirements.txt
python3 -m ml_gcam --help

# or reference it directly
.venv/bin/pip install -r requirements.txt
.venv/bin/python3 -m ml_gcam --help

```

- After activating the environment, you can proceed with the rest of the setup.

### Configuration

Configuration variables can be set one of three ways via the package [python-configuration](https://github.com/tr11/python-configuration) and are used to in the config module to modify behaviors of the entire training and running process:

1. via the system environment

```
# prefix the key with 'ML_GCAM', then separate each nesting with two underscores:

os.environ['ML_GCAM__MODEL__HIDDEN_SIZE'] = 16
config.reload()
assert config.model.hidden_size == 16

```

2. via the `.env` file in the project directory root

copy .env.example and change default values to match your preferences and system:

```cp .env.example .env```

Update before running:
- RESEARCH_DRIVE
- WORKSPACE_DRIVE
- REPO_ROOT
- WANDB API
- TARGETS PATH


3. via .toml files in `ml_gcam/config` directory

- Once modified, the config can be accessed via the `config` object in `ml_gcam.__init__.py`:

```
# from ml_gcam/training/train.py:
from ml_gcam import config

...

emulator = DNN(
    in_size=len(config.data.input_keys),
    hidden_size=int(config.model.hidden_size),
    depth=int(config.model.depth),
    n_heads=n_heads,
    n_features=len(config.data.output_keys),
)

```

### Wandb

We use weights and biases for logging. To use it, you will need to make an account: https://wandb.ai/site

Then set the appropriate config variables:

```
# .env:

WANDB_API_KEY="[replace with api key]"
ML_GCAM__WANDB__ENTITY="wandb username/entity"
ML_GCAM__WANDB__TAGS="test,perfect"
ML_GCAM__WANDB__GROUP="dd_mm_name_is_good"

```


## Dataset Generation

### Sources

| Sampling    | Range        |   Training Scenarios |   Validation Scenarios |   Test Scenarios | New GCAM Samples?   |
|:------------|:-------------|---------------------:|-----------------------:|-----------------:|:--------------------|
| Binary      | x \in {0, 1} |                 3274 |                    409 |              409 | Yes                 |
| Random      | x \in [0, 1] |                 3209 |                    401 |              400 | Yes                 |
| Finite Diff | x \in [0, 1] |                    0 |                      0 |             3650 | Yes                 |

### Inputs

| Input           | Description                                                         | Key    | Interpolated?   |
|:----------------|:--------------------------------------------------------------------|:-------|:----------------|
| Energy demand   | Demand, GDP, and population assumptions                             | energy | Yes             |
| Fossil Fuel     | Costs of oil, natural gas and coal                                  | ff     | Yes             |
| Nuclear         | Capital overnight cost                                              | nuc    | Yes             |
| Solar Storage   | Solar storage capital overnight cost                                | solarS | Yes             |
| Solar Tech      | Concentrating solar-thermal power and photovoltaic technology costs | solarT | Yes             |
| Wind Storage    | Wind storage capital overnight cost                                 | windS  | Yes             |
| Wind Tech       | Wind and wind offshore capital overnight cost                       | windT  | Yes             |
| Backups         | Systems needed to backup solar and wind                             | back   | Yes             |
| Carbon Capture  | Cost of ccs technologies                                            | ccs    | Yes             |
| Electrification | Share of buildings, industries and transport using electricity      | elec   | No              |
| Emissions       | 𝐶𝑂2 emission costs                                                  | emiss  | No              |
| Bioenergy       | Potential costs to adoption of bioenergy                            | bio    | No              |

### Outputs

| Resource   | Metric             | Sector      | Units        | Query                                       |
|:-----------|:-------------------|:------------|:-------------|:--------------------------------------------|
| energy     | demand_elecricity  | building    | EJ           | elec_consumption_by_demand_sector           |
| energy     | demand_elecricity  | industry    | EJ           | elec_consumption_by_demand_sector           |
| energy     | demand_elecricity  | transport   | EJ           | elec_consumption_by_demand_sector           |
| energy     | demand_fuel        | building    | EJ           | final_energy_consumption_by_sector_and_fuel |
| energy     | demand_fuel        | building    | EJ           | final_energy_consumption_by_sector_and_fuel |
| energy     | demand_fuel        | industry    | EJ           | final_energy_consumption_by_sector_and_fuel |
| energy     | demand_fuel        | industry    | EJ           | final_energy_consumption_by_sector_and_fuel |
| energy     | demand_fuel        | transport   | EJ           | final_energy_consumption_by_sector_and_fuel |
| energy     | price              | coal        | 1975$/GJ     | final_energy_prices                         |
| energy     | price              | electricity | 1975$/GJ     | final_energy_prices                         |
| energy     | price              | transport   | 1975$/GJ     | final_energy_prices                         |
| energy     | price              | transport   | 1975$/GJ     | final_energy_prices                         |
| energy     | supply_electricity | biomass     | EJ           | elec_gen_by_subsector                       |
| energy     | supply_electricity | coal        | EJ           | elec_gen_by_subsector                       |
| energy     | supply_electricity | gas         | EJ           | elec_gen_by_subsector                       |
| energy     | supply_electricity | nuclear     | EJ           | elec_gen_by_subsector                       |
| energy     | supply_electricity | oil         | EJ           | elec_gen_by_subsector                       |
| energy     | supply_electricity | other       | EJ           | elec_gen_by_subsector                       |
| energy     | supply_electricity | solar       | EJ           | elec_gen_by_subsector                       |
| energy     | supply_electricity | wind        | EJ           | elec_gen_by_subsector                       |
| energy     | supply_primary     | biomass     | EJ           | primary_energy_consumption_by_region        |
| energy     | supply_primary     | coal        | EJ           | primary_energy_consumption_by_region        |
| energy     | supply_primary     | gas         | EJ           | primary_energy_consumption_by_region        |
| energy     | supply_primary     | nuclear     | EJ           | primary_energy_consumption_by_region        |
| energy     | supply_primary     | oil         | EJ           | primary_energy_consumption_by_region        |
| energy     | supply_primary     | other       | EJ           | primary_energy_consumption_by_region        |
| energy     | supply_primary     | solar       | EJ           | primary_energy_consumption_by_region        |
| energy     | supply_primary     | wind        | EJ           | primary_energy_consumption_by_region        |
| land       | allocation         | biomass     | thousand km2 | aggregated_land_allocation                  |
| land       | allocation         | forest      | thousand km2 | aggregated_land_allocation                  |
| land       | allocation         | grass       | thousand km2 | aggregated_land_allocation                  |
| land       | allocation         | other       | thousand km2 | aggregated_land_allocation                  |
| land       | allocation         | pasture     | thousand km2 | aggregated_land_allocation                  |
| land       | demand             | feed        | Mt           | demand_balances_by_crop_commodity           |
| land       | demand             | food        | Mt           | demand_balances_by_crop_commodity           |
| land       | price              | biomass     | 1975$/GJ     | prices_by_sector                            |
| land       | price              | forest      | 1975$/m3     | prices_by_sector                            |
| land       | production         | biomass     | EJ           | ag_production_by_crop_type                  |
| land       | production         | forest      | billion m3   | ag_production_by_crop_type                  |
| land       | production         | grass       | Mt           | ag_production_by_crop_type                  |
| land       | production         | other       | Mt           | ag_production_by_crop_type                  |
| land       | production         | pasture     | Mt           | ag_production_by_crop_type                  |
| water      | demand             | crops       | km3          | water_withdrawls_by_tech                    |
| water      | demand             | electricity | km3          | water_withdrawls_by_tech                    |

To generate the raw data used in the supervised learning task, runs of [GCAM core](https://github.com/JGCRI/gcam-core) with the specific set of configurations outlined in [Woodward (2023)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022EF003442) are required.
In this study, GCAM core was run 2^12 times, one for each permutation of the 12 input variables.
Reference configuration files are provided in the `ml_gcam/core/configs/*.xml` files.
This process will generate a set of basex files, one for each configuration.

### Interpolation

To complete the interpolation study, these configuration files need to be modified.
Scripts for doing so are available:

```
> python -m ml_gcam --help | grep interpolate
core-interpolate:make-configs  create configs from set of interpolated...
core-interpolate:make-inputs   create paths containing interpolated inputs
core-interpolate:sample        creates a metadata.csv with [--sample]...
```

### Extraction

GCAM core outputs, in the form of basex files, can be extracted to .csv using either the [ModelInterface](https://github.com/JGCRI/modelinterface/) tool, or the python [gcamreader](https://github.com/JGCRI/gcamreader/) package.
Both tools rely on query .xml files to define the data to extract from the basex files.
Reference query files are available in the `ml_gcam/core/queries/*.xml` files.
A typical run to extract a query using `gcamreader` looks something like this:

```
> python3 -m gcamreader local --help
Usage: python -m gcamreader local [OPTIONS]

  query gcam scenario databases

Options:
  -d, --database_path DIRECTORY  path to database file (i.e. parent of *.basex
                                 dir)  [required]
  -q, --query_path FILE          path to xml with queries to run (i.e:
                                 Main_queries.xml)  [required]
  -o, --output_path DIRECTORY    path to output (i.e. where .csv files should
                                 be created)
  -f, --force BOOLEAN            overwrite existing .csv in output path
  --help                         Show this message and exit.

> python3 -m gcamreader local \
 --query_path ml_gcam/core/queries/Main_queries.xml \
 --database_path /path/to/.basex/parent \
 --outpub_path data/query_outputs

...
```

Once extracted, individual queries are aggregated to generate the targets for the emulator.
A scripts for going so are available:
```
> python -m ml_gcam data:create-extracts --help
Usage: python -m ml_gcam data:create-extracts [OPTIONS]

  aggregated raw extract csv from all experiments via sql templates

Options:
  -e, --experiment [binary|interp_hypercube|interp_dgsm]
  -q, --queries [agriculture_prices|electricity_supply|emissions_capture|energy_demand_share_electricity|energy_demand_share_primary|energy_prices|energy_supply_share_electricity|energy_supply_share_primary|land_demand|land_prices|land_supply_allocation|land_supply_production|water_demand|water_consumption]
  -g, --gcamreader-outputs DIRECTORY
                                  directory with the .csv outputs from
                                  gcamreader  [required]
  --save_path DIRECTORY           directory to save {experiment}.csv outputs
                                  (/path/to/data/targets/)
  -f, --force
  --pretend / --no-pretend
  --help                          Show this message and exit.
```

To turn these extracts into an aggregated version for training, run the following command:

```
> python3 -m ml_gcam data:create-targets --help
Usage: python -m ml_gcam data:create-targets [OPTIONS]

  create meta.scenarios table

Options:
  -e, --experiment [binary|interp_hypercube|interp_dgsm]
  --targets_path DIRECTORY        /path/to/targets/
  --scenarios_path FILE           path to scenarios.csv
  -s, --save_path FILE            /path/to/targets/targets.parquet (output
                                  partitioned by experiment and split)
  -f, --force
  --help                          Show this message and exit.
```

This will create a [parquet](https://parquet.apache.org/) file named `targets.parquet` that is used by all downstream tasks for training and validation.

## Training

To train the emulator
Our project uses a command-line interface (CLI) to manage various training loops for machine learning models using PyTorch and Weights & Biases (wandb). Below, you'll find detailed instructions on how to use each command in the training module.

### General Usage

All training loop commands are executed as a module argument to Python and require specific options depending on the task:

`python -m ml_gcam <command> [OPTIONS]`

Commands Overview

- `training:run`: Main training loop.
- `training:sample-size`: Training loop for experimenting with different sample sizes.
- `training:cartesian`: Training loop for experimenting with combination of different sampling strategies.
- `training:sweep-init`: Initial setup for hyperparameter sweeps.
- `training:sweep-run`: Execution of hyperparameter sweeps.

### Example Command
```bash
python -m ml_gcam --no-wandb training:run \
  --targets_path /path/to/targets.parquet \
  --train_source binary \
  --dev_source binary \
  --checkpoint_path /path/to/checkpoint
```

### Command Help

```bash
> python -m ml_gcam training:run --help
Usage: python -m ml_gcam training:run [OPTIONS]

  main training loop

Options:
  -n, --normalization-strategy [z_score|min_max|robust]
                                  how to handle normalization of target before
                                  training  [required]
  -c, --checkpoint_path FILE      /path/to/model/checkpoint/name/
  -d, --dev_source [mixed|binary|interp_hypercube]
                                  name of experiment(s) to use as dev_source
                                  [default: interp_sobol; required]
  -t, --train_source [mixed|interp_hypercube|binary]
                                  name of experiment(s) to use as train_source
                                  [default: binary; required]
  --targets_path FILE             /path/to/targets.parquet  [default: /path/to
                                  /targets.parquet; required]
  --help                          Show this message and exit.
```

```bash
> python -m ml_gcam training:sample-size --help
Usage: python -m ml_gcam training:sample-size [OPTIONS]

  training loop for sample size experiments

Options:
  --targets_path FILE             /path/to/targets.parquet  [default: /path/to
                                  /targets.parquet; required]
  -t, --train_source [mixed|interp_hypercube|binary]
                                  name of experiment(s) to use as train_source
                                  [default: binary; required]
  -d, --dev_source [mixed|interp_hypercube|binary]
                                  name of experiment(s) to use as dev_source
                                  [default: interp_sobol; required]
  -c, --checkpoint_path DIRECTORY
                                  /path/to/model/checkpoint/name/
  -s, --splits FLOAT              samples to use for training, in percent of
                                  train_set  [default: 0.01, 0.02, 0.05,
                                  0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.6, 0.65,
                                  0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0;
                                  required]
  --help                          Show this message and exit.
```

```bash
> python -m ml_gcam training:cartesian --help
Usage: python -m ml_gcam training:cartesian [OPTIONS]

  configure a wandb hyperparameter sweep and make a sweep id

Options:
  --targets_path FILE             /path/to/targets.parquet  [default: /path/to
                                  /targets.parquet; required]
  -t, --train_source [mixed|interp_hypercube|binary]
                                  name of experiment(s) to use as train_source
                                  [default: mixed, interp_hypercube,
                                  binary; required]
  -d, --dev_source [mixed|interp_hypercube|binary]
                                  name of experiment(s) to use as dev_source
                                  [default: mixed, interp_hypercube,
                                  binary; required]
  -c, --checkpoint_path DIRECTORY
                                  /path/to/model/checkpoint/name/
  --help                          Show this message and exit.
```

## Evaluation

We picked r2 scores as the more important evaluation metric for these experiment.
During training, some aggregate statistics are logged to the console and to wandb if enabled.
To generate these metrics from a saved model checkpoint, use the following commands:

```bash
Usage: python -m ml_gcam evaluate:sample-size [OPTIONS]

  load and create r2 scores for training size sweep

Options:
  -c, --checkpoint_path DIRECTORY
                                  /path/to/model/checkpoints/for-sample-size-sweep/
                                  [required]
  -t, --train_source [mixed|binary|interp_hypercube]
                                  [required]
  -o, --save_path FILE            path to save score.csv  [required]
  -f, --force                     forces the --save_path to overwrite existing
                                  data
  --targets_path FILE             /path/to/targets.parquet  [default: /path/to
                                  /targets.parquet; required]
  --help                          Show this message and exit.
```

```bash
> python -m ml_gcam evaluate:cartesian --help
Usage: python -m ml_gcam evaluate:cartesian [OPTIONS]

  load and create r2 scores for cartesian sweep

Options:
  -c, --checkpoint_path DIRECTORY
                                  /path/to/model/checkpoints/for-cartesian-sweep/
                                  [required]
  -o, --save_path FILE            path to save score.csv  [required]
  -f, --force                     forces the --save_path to overwrite existing
                                  data
  --targets_path FILE             /path/to/targets.parquet  [default:
                                  /path/to/targets.parquet; required]
  --help                          Show this message and exit.``

```
## Plots and Tables

```bash
> python -m ml_gcam table:cartesian --help
Usage: python -m ml_gcam table:cartesian [OPTIONS]

  display the cartesian product of training x dev sets

Options:
  -f, --force                     forces the --save_path to overwrite existing
                                  data
  -o, --save_path FILE            path to save figure.png  [required]
  -t, --table_format              [latex|markdown]
  -s, --score_path FILE           /path/to/cartesian/scores.csv  [required]
  --help                          Show this message and exit.
```

```bash
> python -m ml_gcam plot:sample-size --help
Usage: python -m ml_gcam plot:sample-size [OPTIONS]

  plot r2 vs. training size from previously generated data

Options:
  -f, --force                     forces the --save_path to overwrite existing
                                  data
  -o, --save_path FILE            path to save figure.png  [required]
  -a, --aggregation [median|mean|above_0_9|above_0_95|overall]
  -s, --score_path FILE           /path/to//sample-size/scores.csv  [required]
  --help                          Show this message and exit.
```
