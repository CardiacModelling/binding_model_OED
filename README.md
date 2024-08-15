# Code for generating results and plotting
The code in this repo generates all synthetic data and figures included in the paper entitled "Optimising experimental designs for model selection of ion channel drug binding mechanisms".

### Structure
- data: Contains fits to data obtained in the Lei et al. paper "The impact of uncertainty in hERG binding mechanism on in silico predictions of drug-induced proarrhythmic risk". This is used for the initial parameterisations of drug-binding models.
- methods: Core python code
- figures: Figures included in the paper
- protocols: Contains the milnes protocol .mmt file
- src: Core source code including bash scripts and python files for generating synthetic data, fitting models, optimising protocols, and plotting figures 

### Running code
All data-generating, model-fitting, and OED code can be run by submitting the bash script via `./src/run_full_grid.sh`

On completion, plots can be generated via `./src/run_full_grid_plotting.sh`
