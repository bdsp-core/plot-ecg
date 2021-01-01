This script [tensorizes](https://github.com/aguirre-lab/ml4c3/wiki/Tensorization) ECG XML files to HD5 files named by that patient's MRN, then plots all ECGs.

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

You can convert ECGs as PDFs (`--ext pdf`) or PNGs (`--ext png`).