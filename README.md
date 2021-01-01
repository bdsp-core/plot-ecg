## About
This script converts ECG XML files to HD5 files named by MRN, then plots all ECGs as either PDFs or PNGs. The plots look like clinical 12-lead ECGs.

Developed by [@erikr](https://github.com/erikr) and the [`ml4c3`](https://github.com/aguirre-lab/ml4c3) team.

## Setup
1. Clone repo
    ```
    git clone git@github.com:aguirre-lab/plot-ecg.git
    ```

1. Set up the environment
    ```
    conda env create -f environment.yml
    conda activate plot-ecg
    ```

1. Run the script
    ```
    python ./src/main.py \
    --xml  /path/to/existing/xmls \
    --hd5  /path/to/new/hd5s \
    --plot /path/to/new/plots
    --ext  pdf
    ```

    ![](plot.gif)

    > Data shown in the above recording are deidentified.

    You can convert ECGs as PDFs (`--ext pdf`) or PNGs (`--ext png`).

## Disclaimer
This repo is *not* actively maintained and does not come with any guarantees for compatibility, correctness, or performance.
Issues, bug reports, and feature requests will not be reviewed. Please feel free to fork this repo for your own purposes.