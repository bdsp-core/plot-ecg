This script tensorizes ECG XML files to HD5, then plots them.

1. Clone repo
    ```
    git clone git@github.com:aguirre-lab/plot-ecg.git
    ```

1. Set up the environment
    ```
    conda env create -f environment.yml
    ```

1. Run the script
    ```
    python ./src/main.py \
    --xml_folder /path/to/existing/xmls \
    --hd5_folder /path/to/new/hd5s \
    --plot_folder /path/to/new/plots
    ```