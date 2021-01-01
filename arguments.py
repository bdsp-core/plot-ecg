# Imports: standard library
import os
import sys
import logging
import argparse
import datetime
import operator
import multiprocessing
from typing import Dict

# Imports: third party
import numpy as np

# Imports: first party
from ml4c3.logger import load_config
from definitions.models import BottleneckType
from tensormap.TensorMap import TensorMap, update_tmaps

BOTTLENECK_STR_TO_ENUM = {
    "flatten_restructure": BottleneckType.FlattenRestructure,
    "global_average_pool": BottleneckType.GlobalAveragePoolStructured,
    "variational": BottleneckType.Variational,
}

# pylint: disable=import-outside-toplevel, unnecessary-comprehension
# pylint: disable=too-many-lines, unused-variable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(
        title="ml4c3 recipes",
        description="Select one of the following recipes: \n"
        "\t * train: ADD DESCRIPTION. \n"
        "\t * train_keras_logreg: ADD DESCRIPTION. \n"
        "\t * train_sklearn_logreg: ADD DESCRIPTION. \n"
        "\t * train_sklearn_svm: ADD DESCRIPTION. \n"
        "\t * train_sklearn_randomforest: ADD DESCRIPTION. \n"
        "\t * train_sklearn_xgboost: ADD DESCRIPTION. \n"
        "\t * train_simclr: ADD DESCRIPTION. \n"
        "\t * infer: ADD DESCRIPTION. \n"
        "\t * hyperoptimize: ADD DESCRIPTION. \n"
        "\t * build: ADD DESCRIPTION. \n"
        "\t * explore: ADD DESCRIPTION. \n"
        "\t * explore_icu: ADD DESCRIPTION. \n"
        "\t * plot_ecg: ADD DESCRIPTION. \n"
        "\t * tensorize_ecg: ADD DESCRIPTION. \n"
        "\t * pull_edw: ADD DESCRIPTION. \n"
        "\t * tensorize_icu_no_edw_pull: ADD DESCRIPTION. \n"
        "\t * tensorize_icu: ADD DESCRIPTION. \n"
        "\t * tensorize_sts: ADD DESCRIPTION. \n"
        "\t * assess_icu_coverage: ADD DESCRIPTION. \n"
        "\t * check_icu_structure: ADD DESCRIPTION. \n"
        "\t * pre_tensorize_explore: ADD DESCRIPTION. \n"
        "\t * match_patient_bedmaster: ADD DESCRIPTION. \n"
        "\t * extract_ecg_features: ADD DESCRIPTION. \n",
        dest="recipe",
    )

    # Tensor Map arguments
    tensormaps_parser = argparse.ArgumentParser(add_help=False)
    tensormaps_parser.add_argument("--input_tensors", default=[], nargs="+")
    tensormaps_parser.add_argument("--output_tensors", default=[], nargs="+")
    tensormaps_parser.add_argument(
        "--tensor_maps_in",
        default=[],
        help="Do not set this directly. Use input_tensors",
    )
    tensormaps_parser.add_argument(
        "--tensor_maps_out",
        default=[],
        help="Do not set this directly. Use output_tensors",
    )

    # Input and Output files and directories
    io_parser = argparse.ArgumentParser(add_help=False)
    io_parser.add_argument(
        "--patient_csv",
        help="Path to CSV with Sample IDs to restrict tensor paths",
    )
    io_parser.add_argument(
        "--mrn_column_name",
        help="Name of MRN column in patient_csv to look for",
    )
    io_parser.add_argument(
        "--tensors",
        action="append",
        nargs="+",
        help="Path to folder containing hd5 files or path to csv file."
        "Specify multiple data sources by repeating this argument."
        "In the data object passed to tensor_from_file functions,"
        "csv files are keyed by the filename by default. To key"
        "csv files by another name, provide the name as a second"
        "argument after the path to the csv file, e.g."
        "--tensors /path/to/data.csv data-name",
    )
    io_parser.add_argument(
        "--output_folder",
        default="./recipes-output",
        help="Path to output folder for recipes.py runs.",
    )
    io_parser.add_argument(
        "--model_file",
        help="Path to a saved model architecture and weights (hd5).",
    )
    io_parser.add_argument(
        "--model_files",
        nargs="*",
        default=[],
        help="List of paths to saved model architectures and weights (hd5).",
    )
    io_parser.add_argument(
        "--donor_layers",
        help=(
            "Path to a model file (hd5) which will be loaded by layer, useful for"
            " transfer learning."
        ),
    )
    io_parser.add_argument(
        "--remap_layer",
        action="append",
        nargs=2,
        help="For transfer layer, manually remap layer from pretrained model to layer"
        " in new model. For example: --rename_layer pretrained_layer_name "
        "new_layer_name. Layers are remapped using this argument one at a time, "
        "repeat for multiple layers.",
    )
    io_parser.add_argument(
        "--freeze_donor_layers",
        action="store_true",
        help="Whether to freeze the layers from donor_layers.",
    )

    # Run specific and debugging arguments
    run_parser = argparse.ArgumentParser(add_help=False)
    cache_parser = run_parser.add_mutually_exclusive_group(required=False)
    cache_parser.add_argument(
        "--cache",
        dest="cache",
        action="store_true",
        help="Enable caching of tf.data.Dataset, on by default.",
    )
    cache_parser.add_argument(
        "--cache_off",
        dest="cache",
        action="store_false",
        help="Disable caching of tf.data.Dataset.",
    )
    cache_parser.set_defaults(cache=True)
    run_parser.add_argument(
        "--random_seed",
        default=2021,
        type=int,
        help="Random seed to use throughout run.  Always use np.random.",
    )
    run_parser.add_argument(
        "--plot_mode",
        default="clinical",
        choices=["clinical", "full"],
        help="ECG view to plot.",
    )
    run_parser.add_argument(
        "--logging_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=(
            "Logging level. Overrides any configuration given in the logging"
            " configuration file."
        ),
    )
    run_parser.add_argument(
        "--image_ext",
        default=".pdf",
        choices=[".pdf", ".eps", ".svg", ".png"],
        help="File format extension to save images as. Includes leading period.",
    )
    run_parser.add_argument(
        "--num_workers",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Number of workers to use for every tensor generator.",
    )
    run_parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Changes behavior in code to facilitate debugging.",
    )

    # ECG Tensorize arguments
    ecg_tensorization_parser = subparser.add_parser(
        name="tensorize_ecg",
        description="TODO",
        parents=[io_parser, run_parser],
    )
    ecg_tensorization_parser.add_argument(
        "--bad_xml_dir",
        default=os.path.expanduser("~/bad-xml"),
        help="Path to directory to store XML files that fail tensorization.",
    )
    ecg_tensorization_parser.add_argument(
        "--bad_hd5_dir",
        default=os.path.expanduser("~/bad-hd5"),
        help="Path to directory to store HD5 files that fail tensorization.",
    )
    ecg_tensorization_parser.add_argument(
        "--xml_folder",
        help="Path to folder of XMLs of ECG data.",
    )

    # ICU modes common arguments
    icu_parser = argparse.ArgumentParser(add_help=False)
    icu_parser.add_argument(
        "--edw",
        default="/media/ml4c3/edw/",
        help="Directory to save or load EDW CSV files.",
    )
    icu_parser.add_argument(
        "--bedmaster",
        default="/media/lm4-bedmaster/",
        help="Directory containing Bedmaster .mat files.",
    )
    icu_parser.add_argument(
        "--alarms",
        default="/media/ml4c3/bedmaster_alarms/",
        help="Directory containing Bedmaster alarms CSV files.",
    )
    icu_parser.add_argument(
        "--cohort_query",
        help="Path to SQL file with query to define a cohort of patients by MRNs and "
        "optionally CSNs.",
    )
    icu_parser.add_argument(
        "--adt",
        help="Path to save or load ADT table CSV file.",
    )
    icu_parser.add_argument(
        "--xref",
        default=os.path.expanduser("~/xref.csv"),
        help="Path to CSV file that links MRN, CSN, and Bedmaster file paths. "
        "If this file does not exist, it is created in Step 2. If this "
        "file exists, it is loaded and used in Step 2 and a new xref.csv "
        "is not created and saved.",
    )
    icu_parser.add_argument(
        "--departments",
        nargs="+",
        help="List of department names for which to process patient data.",
    )
    icu_parser.add_argument(
        "--staging_dir",
        help="Directory to store temp files created during tensorization pipeline.",
    )
    icu_parser.add_argument(
        "--staging_batch_size",
        type=int,
        default=1,
        help="Number of patients to process locally before being moved to final "
        "location.",
    )
    icu_parser.add_argument(
        "--mrn_start_index",
        type=int,
        default=0,
        help="MRN of first patient in ADT table to process, inclusive.",
    )
    icu_parser.add_argument(
        "--mrn_end_index",
        type=int,
        help="MRN of last patient in ADT table to process, exclusive.",
    )
    icu_parser.add_argument(
        "--adt_start_index",
        type=int,
        default=0,
        help="Index of first MRN,CSN row in ADT table to process, inclusive.",
    )
    icu_parser.add_argument(
        "--adt_end_index",
        type=int,
        help="Index of last MRN,CSN row in ADT table to process, exclusive.",
    )
    icu_parser.add_argument(
        "--start_time",
        help="Only use patients admitted after this date, YYYY-MM-DD format.",
    )
    icu_parser.add_argument(
        "--end_time",
        help="Only use patients admitted before this date, YYYY-MM-DD format.",
    )
    icu_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing EDW and hd5 files during tensorization.",
    )
    icu_parser.add_argument(
        "--allow_one_source",
        action="store_true",
        help="If this parameter is set, patients with just one type of data "
        "will be tensorized.",
    )

    # Pull ADT table
    pull_adt_parser = subparser.add_parser(
        name="pull_adt",
        description="TODO",
        parents=[run_parser, io_parser, icu_parser],
    )

    # Pull EDW data for ICU tensorization
    pull_edw_parser = subparser.add_parser(
        name="pull_edw",
        description="TODO",
        parents=[run_parser, io_parser, icu_parser],
    )

    # ICU tensorization with existing EDW data
    icu_tensorization_parser = subparser.add_parser(
        name="tensorize_icu_no_edw_pull",
        description="TODO",
        parents=[run_parser, io_parser, icu_parser],
    )

    # End to end ICU tensorization
    end_to_end_icu_tensorization_parser = subparser.add_parser(
        name="tensorize_icu",
        description="TODO",
        parents=[run_parser, io_parser, icu_parser],
    )

    # Model Architecture Parameters
    model_parser = argparse.ArgumentParser(add_help=False)
    model_parser.add_argument(
        "--conv_type",
        default="conv",
        choices=["conv", "separable", "depth"],
        help="Type of convolutional layer used in conv, residual, and dense blocks",
    )
    model_parser.add_argument(
        "--conv_blocks",
        nargs="*",
        type=int,
        help=(
            "Number of filters to use in all convolutional layers for each "
            "convolutional block."
        ),
    )
    model_parser.add_argument(
        "--conv_block_size",
        type=int,
        help="Number of convolutional layers within a convolutional block.",
    )
    model_parser.add_argument(
        "--conv_block_layer_order",
        nargs=4,
        default=["convolution", "normalization", "activation", "dropout"],
        choices=["convolution", "normalization", "activation", "dropout"],
        help="TODO",
    )
    model_parser.add_argument(
        "--residual_blocks",
        nargs="*",
        type=int,
        help=(
            "Number of filters to use in all convolutional layers for each residual "
            "block. Original residual block paper: https://arxiv.org/abs/1512.03385"
        ),
    )
    model_parser.add_argument(
        "--residual_block_size",
        type=int,
        help="Number of convolutional layers within a residual block.",
    )
    model_parser.add_argument(
        "--residual_block_layer_order",
        nargs=4,
        default=["convolution", "normalization", "activation", "dropout"],
        choices=["convolution", "normalization", "activation", "dropout"],
        help="TODO",
    )
    model_parser.add_argument(
        "--dense_blocks",
        nargs="*",
        type=int,
        help=(
            "Number of filters to use in all convolutional layers for each dense block."
            " Original dense block paper: https://arxiv.org/abs/1608.06993"
        ),
    )
    model_parser.add_argument(
        "--dense_block_size",
        type=int,
        help="Number of convolutional layers within a dense block.",
    )
    model_parser.add_argument(
        "--dense_block_layer_order",
        nargs=4,
        default=["convolution", "normalization", "activation", "dropout"],
        choices=["convolution", "normalization", "activation", "dropout"],
        help="TODO",
    )
    model_parser.add_argument(
        "--conv_x",
        nargs="*",
        type=int,
        help=(
            "X dimension of convolutional kernel. Kernel dimensions are specified per "
            "conv_block, residual_block, and dense_block. Kernel dimensions are "
            "repeated if the number of kernels specified is less than the number of "
            "blocks."
        ),
    )
    model_parser.add_argument(
        "--conv_y",
        nargs="*",
        type=int,
        help="Y dimension of convolutional kernel. See --conv_x.",
    )
    model_parser.add_argument(
        "--conv_z",
        nargs="*",
        type=int,
        help="Z dimension of convolutional kernel. See --conv_x.",
    )
    model_parser.add_argument(
        "--conv_padding",
        default="same",
        help="Valid or same border padding on the convolutional layers.",
    )
    model_parser.add_argument(
        "--pool_type",
        nargs="?",
        choices=["max", "average"],
        help=(
            "Type of pooling layers inserted between conv_layers, residual_blocks, and "
            "dense_blocks."
        ),
    )
    model_parser.add_argument(
        "--pool_x",
        type=int,
        help="Pooling size in the x-axis, if 1 no pooling will be performed.",
    )
    model_parser.add_argument(
        "--pool_y",
        type=int,
        help="Pooling size in the y-axis, if 1 no pooling will be performed.",
    )
    model_parser.add_argument(
        "--pool_z",
        type=int,
        help="Pooling size in the z-axis, if 1 no pooling will be performed.",
    )
    model_parser.add_argument(
        "--bottleneck_type",
        type=str,
        default=list(BOTTLENECK_STR_TO_ENUM)[0],
        choices=list(BOTTLENECK_STR_TO_ENUM),
    )
    model_parser.add_argument(
        "--dense_layers",
        nargs="*",
        type=int,
        help="List of number of hidden units in dense (fully connected) layers.",
    )
    model_parser.add_argument(
        "--activation_layer",
        nargs="?",
        help=(
            "Type of activation layer after dense (fully connected) or convolutional "
            "layers."
        ),
    )
    model_parser.add_argument(
        "--normalization_layer",
        nargs="?",
        choices=["batch_norm", "layer_norm", "instance_norm", "poincare_norm"],
        help=(
            "Type of normalization layer after dense (fully connected) or convolutional"
            " layers."
        ),
    )
    model_parser.add_argument(
        "--spatial_dropout",
        default=0.0,
        type=float,
        help="Dropout rate of convolutional kernels; must be in [0, 1].",
    )
    model_parser.add_argument(
        "--dense_dropout",
        default=0.0,
        type=float,
        help="Dropout rate of dense (fully connected) layers; must be in [0, 1].",
    )
    model_parser.add_argument(
        "--dense_layer_order",
        nargs=4,
        default=["dense", "normalization", "activation", "dropout"],
        choices=["dense", "normalization", "activation", "dropout"],
        help=(
            "Order of dense (fully connected), normalization, activation, and dropout "
            "layers."
        ),
    )
    model_parser.add_argument(
        "--nest_model",
        nargs=2,
        action="append",
        help="Embed a nested model ending at the specified layer before the bottleneck"
        " layer of the current model. Repeat this argument to embed multiple models."
        " For example --nest_model /path/to/model_weights.h5 embed_layer",
    )
    model_parser.add_argument(
        "--save_coefficients",
        action="store_true",
        help="Save model coefficients to CSV file",
    )
    model_parser.add_argument(
        "--top_features_to_plot",
        type=int,
        help="Number of features to plot in features coefficients plot.",
    )

    # Training Parameters
    training_parser = argparse.ArgumentParser(add_help=False)
    training_parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs.",
    )
    training_parser.add_argument(
        "--batch_size",
        type=int,
        help="Mini batch size for stochastic gradient descent algorithms.",
    )
    training_parser.add_argument(
        "--train_csv",
        help="Path to CSV with Sample IDs to reserve for training.",
    )
    training_parser.add_argument(
        "--valid_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for validation. Takes precedence"
            " over valid_ratio."
        ),
    )
    training_parser.add_argument(
        "--test_csv",
        help=(
            "Path to CSV with Sample IDs to reserve for testing. Takes precedence over"
            " test_ratio."
        ),
    )
    training_parser.add_argument(
        "--valid_ratio",
        default=0.2,
        type=float,
        help=(
            "Rate of training tensors to save for validation must be in [0.0, 1.0]. If"
            " any of train/valid/test csv is specified, split by ratio is applied on"
            " the remaining tensors after reserving tensors given by csvs. If not"
            " specified, default 0.2 is used. If default ratios are used with"
            " train_csv, some tensors may be ignored because ratios do not sum to 1."
        ),
    )
    training_parser.add_argument(
        "--test_ratio",
        default=0.1,
        type=float,
        help=(
            "Rate of training tensors to save for testing must be in [0.0, 1.0]. If any"
            " of train/valid/test csv is specified, split by ratio is applied on the"
            " remaining tensors after reserving tensors given by csvs. If not"
            " specified, default 0.1 is used. If default ratios are used with"
            " train_csv, some tensors may be ignored because ratios do not sum to 1."
        ),
    )
    training_parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate during training.",
    )
    training_parser.add_argument(
        "--learning_rate_patience",
        default=8,
        type=int,
        help="Number of epochs without validation loss improvement to wait before"
        " reducing learning rate by multiplying by the learning_rate_reduction scale"
        " factor.",
    )
    training_parser.add_argument(
        "--learning_rate_reduction",
        default=0.5,
        type=float,
        help="Scale factor to reduce learning rate by.",
    )
    training_parser.add_argument(
        "--mixup_alpha",
        default=0,
        type=float,
        help="If non-zero, mixup batches with this alpha parameter for mixup.",
    )
    training_parser.add_argument(
        "--patience",
        default=24,
        type=int,
        help=(
            "Early Stopping parameter: Maximum number of epochs to run without"
            " validation loss improvements."
        ),
    )
    training_parser.add_argument(
        "--optimizer",
        help="Optimizer for model training",
    )
    training_parser.add_argument(
        "--learning_rate_schedule",
        choices=["triangular", "triangular2"],
        help="Adjusts learning rate during training.",
    )
    training_parser.add_argument(
        "--l1",
        default=0.0,
        type=float,
        help="L1 value for regularizing the kernel and bias of each layer.",
    )
    training_parser.add_argument(
        "--l2",
        default=0.0,
        type=float,
        help="L2 value for regularizing the kernel and bias of each layer.",
    )

    # Train Shallow arguments
    train_shallow_parser = argparse.ArgumentParser(add_help=False)
    train_shallow_parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="Number of estimators",
    )
    train_shallow_parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of tree-based classifier",
    )
    train_shallow_parser.add_argument(
        "--c",
        type=float,
        default=1e9,
        help="Inverse of regularization strength; must be a positive float. "
        "Smaller values specify stronger regularization",
    )
    train_shallow_parser.add_argument(
        "--min_samples_split",
        type=int,
        default=5,
        help="",
    )
    train_shallow_parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=8,
        help="",
    )
    train_shallow_parser.add_argument(
        "--gamma",
        default=0.0,
        type=float,
        help="Minimum loss reduction required to make a further partition on a "
        "leaf node of the tree.",
    )

    # Training modes parsers
    train_parser = subparser.add_parser(
        name="train",
        description="TODO",
        parents=[
            model_parser,
            training_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    train_keras_logreg_parser = subparser.add_parser(
        name="train_keras_logreg",
        description="TODO",
        parents=[
            training_parser,
            train_shallow_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    train_sklearn_logreg_parser = subparser.add_parser(
        name="train_sklearn_logreg",
        description="TODO",
        parents=[
            training_parser,
            train_shallow_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    train_svm_parser = subparser.add_parser(
        name="train_sklearn_svm",
        description="TODO",
        parents=[
            training_parser,
            train_shallow_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    train_randomforest_parser = subparser.add_parser(
        name="train_sklearn_randomforest",
        description="TODO",
        parents=[
            training_parser,
            train_shallow_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    train_xgboost_parser = subparser.add_parser(
        name="train_sklearn_xgboost",
        description="TODO",
        parents=[
            training_parser,
            train_shallow_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    train_simclr_parser = subparser.add_parser(
        name="train_simclr",
        description="TODO",
        parents=[
            model_parser,
            training_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    hyperoptimize_parser = subparser.add_parser(
        name="hyperoptimize",
        description="TODO",
        parents=[
            model_parser,
            training_parser,
            train_shallow_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )

    # Hyperoptimize arguments
    hyperoptimize_parser.add_argument(
        "--max_parameters",
        default=9000000,
        type=int,
        help="Maximum number of trainable parameters in a model during "
        "hyperoptimization.",
    )
    hyperoptimize_parser.add_argument(
        "--max_evals",
        default=16,
        type=int,
        help=(
            "Maximum number of models for the hyperparameter optimizer to evaluate"
            " before returning."
        ),
    )
    hyperoptimize_parser.add_argument(
        "--make_training_plots",
        action="store_true",
        help="Whether or not to plot calibration, ROC and PR curvese from the "
        "training set.",
    )
    hyperoptimize_parser.add_argument(
        "--hyperoptimize_config_file",
        type=str,
        help="Full path to to .json file with the parameters to hyperoptimize. You can "
        "use the script generate_hyperoptimize_json.py to create it.",
    )

    # Explore arguments
    explore_parser = subparser.add_parser(
        name="explore",
        description="TODO",
        parents=[io_parser, run_parser, tensormaps_parser, training_parser],
    )
    explore_parser.add_argument(
        "--explore_disable_saving_output",
        action="store_true",
        help="Disable saving outputs from explore: "
        "histograms, summary stats, and tensors.",
    )
    explore_parser.add_argument(
        "--explore_export_error",
        action="store_true",
        help="Export error_type in tensors_all_*.csv generated by explore.",
    )
    explore_parser.add_argument(
        "--explore_export_fpath",
        action="store_true",
        help="Export path to HD5 in tensors_all_*.csv generated by explore.",
    )
    explore_parser.add_argument(
        "--explore_export_generator",
        action="store_true",
        help="Export generator (e.g. train, valid, or test split) in "
        "tensors_all_*.csv generated by explore.",
    )
    explore_parser.add_argument(
        "--explore_stratify_label",
        help=(
            "TensorMap or column name of value in CSV to stratify distribution around,"
            " e.g. mortality. Optional."
        ),
    )
    explore_parser.add_argument(
        "--source_name",
        default="ecg",
        help=(
            "Name of source dataset at tensors, e.g. ECG. "
            "Adds contextual detail to summary CSV and plots."
        ),
    )
    explore_parser.add_argument(
        "--join_tensors",
        nargs="+",
        help=(
            "TensorMap or column name in csv of value in tensors used in join with"
            " reference. Can be more than 1 join value."
        ),
    )
    explore_parser.add_argument(
        "--time_tensor",
        help=(
            "TensorMap or column name in csv of value in tensors to perform time"
            " cross-ref on. Time cross referencing is optional."
        ),
    )
    explore_parser.add_argument(
        "--reference_tensors",
        help="Either a csv or directory of hd5 containing a reference dataset.",
    )
    explore_parser.add_argument(
        "--reference_name",
        default="Reference",
        help=(
            "Name of dataset at reference, e.g. STS. "
            "Adds contextual detail to summary CSV and plots."
        ),
    )
    explore_parser.add_argument(
        "--reference_join_tensors",
        nargs="+",
        help=(
            "TensorMap or column name in csv of value in reference used in join in"
            " tensors. Can be more than 1 join value."
        ),
    )
    explore_parser.add_argument(
        "--reference_start_time_tensor",
        action="append",
        nargs="+",
        help=(
            "TensorMap or column name in csv of start of time window in reference."
            " Define multiple time windows by using this argument more than once. The"
            " number of time windows must match across all time window arguments. An"
            " integer can be provided as a second argument to specify an offset to the"
            " start time. e.g. tStart -30"
        ),
    )
    explore_parser.add_argument(
        "--reference_end_time_tensor",
        action="append",
        nargs="+",
        help=(
            "TensorMap or column name in csv of end of time window in reference. Define"
            " multiple time windows by using this argument more than once. The number"
            " of time windows must match across all time window arguments. An integer"
            " can be provided as a second argument to specify an offset to the end"
            " time. e.g. tEnd 30"
        ),
    )
    explore_parser.add_argument(
        "--window_name",
        action="append",
        help=(
            "Name of time window. By default, name of window is index of window."
            " Define multiple time windows by using this argument multiple times."
            " The number of time windows must match across all time window arguments."
        ),
    )
    explore_parser.add_argument(
        "--order_in_window",
        action="append",
        choices=["newest", "oldest", "random"],
        help=(
            "If specified, exactly --number_in_window rows with join tensor are used in"
            " time window. Defines which source tensors in a time series to use in time"
            " window. Define multiple time windows by using this argument more than"
            " once. The number of time windows must match across all time window"
            " arguments."
        ),
    )
    explore_parser.add_argument(
        "--number_per_window",
        type=int,
        default=1,
        help=(
            "Minimum number of rows with join tensor to use in each time window. "
            "By default, 1 tensor is used for each window."
        ),
    )
    explore_parser.add_argument(
        "--match_any_window",
        action="store_true",
        help=(
            "If specified, join tensor does not need to be found in every time window."
            " Join tensor needs only be found in at least 1 time window. Default only"
            " use rows with join tensor that appears across all time windows."
        ),
    )
    explore_parser.add_argument(
        "--reference_labels",
        nargs="+",
        help=(
            "TensorMap or column name of values in csv to report distribution on, e.g."
            " mortality. Label distribution reporting is optional. Can list multiple"
            " labels to report."
        ),
    )

    # Check structure parser
    check_structure_parser = subparser.add_parser(
        "check_structure",
        description="Verify EDW and Bedmaster files and directories structure "
        "before tensorizing.",
        parents=[icu_parser],
    )
    check_structure_parser.add_argument(
        "--check_edw",
        action="store_true",
        help="If this parameter is set, the EDW files and directory structure "
        "are verified.",
    )
    check_structure_parser.add_argument(
        "--check_bedmaster",
        action="store_true",
        help="If this parameter is set, the Bedmaster files and directory "
        "structure are verified.",
    )

    # Match patient parser
    match_patient_parser = subparser.add_parser(
        "match_patient_bedmaster",
        description="Create a cross-reference "
        "table between the available Bedmaster files and EDW data from an ADT table.",
        parents=[io_parser, run_parser, icu_parser],
    )

    # Pre tensorize summary parser
    pre_tensorize_explorer_parser = subparser.add_parser(
        "pre_tensorize_explore",
        description="Calculate summary statistics before tensorizing.",
        parents=[io_parser, run_parser, icu_parser],
    )
    pre_tensorize_explorer_parser.add_argument(
        "--summary_stats_base_name",
        default="pre_tensorize",
        help="Base name of the summary stats .csv files. " "By default: pre_tensorize",
    )
    pre_tensorize_explorer_parser.add_argument(
        "--signals",
        nargs="+",
        help="List of Bedmaster signals to calculate their summary_statistics. To "
        "calculate statistics for all signals select 'all'. It always "
        "calculates statistics for all EDW signals.",
    )
    pre_tensorize_explorer_parser.add_argument(
        "--detailed_bedmaster",
        action="store_true",
        help="Generate detailed statistics for Bedmaster, like time "
        "irregularities frequence. This option can take some time"
        "to complete.",
    )
    pre_tensorize_explorer_parser.add_argument(
        "--no_xref",
        action="store_true",
        help="Don't cross-reference files. Enable along with --detailed_bedmaster "
        "to get statistics from bedmaster files without needing "
        "an edw and xref path.",
    )

    # Visualizer parser
    visualizer_parser = subparser.add_parser(
        name="visualize",
        description="Start a web server to visualize the generated "
        "HD5 files on the given directory.",
        parents=[io_parser, run_parser],
    )
    visualizer_parser.add_argument(
        "--port",
        "-p",
        default="8050",
        help="Specify the port where the server will run. Default: 8050",
    )
    visualizer_parser.add_argument(
        "--address",
        "-a",
        default="0.0.0.0",
        help="Specify the address where the server will run. "
        "Default: 0.0.0.0 (localhost)",
    )
    visualizer_parser.add_argument(
        "--options_file",
        "-o",
        help="YAML file with options for the visualizer. Default: None",
    )

    # Assess coverage parser
    assess_coverage_parser = subparser.add_parser(
        "assess_coverage",
        description="Assess Bedmaster and HD5 coverage by means of MRNs and CSNs.",
        parents=[io_parser, run_parser, icu_parser],
    )
    assess_coverage_parser.add_argument(
        "--event_column",
        type=str,
        help="Name of the event column (if exists) in --cohort_query/--cohort_csv.",
    )
    assess_coverage_parser.add_argument(
        "--time_column",
        type=str,
        help="Name of the event time column (if exists) in "
        "--cohort_query/--cohort_csv.",
    )
    assess_coverage_parser.add_argument(
        "--cohort_csv",
        type=str,
        help="Full path of the .csv file containing a list of patients. "
        "If --cohort_query is set, this parameter will be ignored.",
    )
    assess_coverage_parser.add_argument(
        "--count",
        action="store_true",
        help="Count the number of unique rows (events) in --cohort_query/--cohort_csv.",
    )

    # ECG feature extraction parser
    ecg_features_parser = subparser.add_parser(
        "extract_ecg_features",
        description="Extract basic ECG features from hd5 files.",
        parents=[io_parser, run_parser, tensormaps_parser],
    )
    ecg_features_parser.add_argument(
        "--r_method",
        type=str,
        default="neurokit",
        help="The algorithm to be used for R-peak detection. Can be one "
        "of neurokit (default), pantompkins1985, hamilton2002, "
        "christov2004, gamboa2008, elgendi2010, engzeemod2012 "
        "or kalidas2017.",
    )
    ecg_features_parser.add_argument(
        "--wave_method",
        type=str,
        default="dwt",
        help="Can be one of dwt (default) for discrete wavelet transform "
        "or cwt for continuous wavelet transform.",
    )

    # Additional subparsers
    infer_parser = subparser.add_parser(
        name="infer",
        description="TODO",
        parents=[
            model_parser,
            training_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    plot_parser = subparser.add_parser(
        name="plot_ecg",
        description="TODO",
        parents=[
            model_parser,
            training_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    build_parser = subparser.add_parser(
        name="build",
        description="TODO",
        parents=[
            model_parser,
            training_parser,
            io_parser,
            run_parser,
            tensormaps_parser,
        ],
    )
    args = parser.parse_args()
    _process_args(args)
    return args


def _process_args(args: argparse.Namespace):
    now_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    args_file = os.path.join(args.output_folder, "arguments_" + now_string + ".txt")
    command_line = f"\n./scripts/run.sh {' '.join(sys.argv)}\n"
    if not os.path.exists(os.path.dirname(args_file)):
        os.makedirs(os.path.dirname(args_file))
    with open(args_file, "w") as f:
        f.write(command_line)
        for k, v in sorted(args.__dict__.items(), key=operator.itemgetter(0)):
            f.write(k + " = " + str(v) + "\n")
    load_config(
        args.logging_level,
        args.output_folder,
        "log_" + now_string,
    )

    tensors = []
    if "tensors" in args and args.tensors is not None:
        for tensor in args.tensors:
            if len(tensor) == 2:
                tensors.append(tuple(tensor))
            elif len(tensor) == 1:
                tensors.append(tensor[0])
            else:
                raise ValueError(
                    f"Tensors must be a path or a path and a name, "
                    f"instead got: {tensor}",
                )
        args.tensors = tensors if len(tensors) > 1 else tensors[0]

    # Create list of names of all needed TMaps
    if "input_tensors" in args and args.recipe != "explore_icu":
        needed_tmaps_names = args.input_tensors + args.output_tensors

        # Update dict of tmaps to include all needed tmaps
        tmaps: Dict[str, TensorMap] = {}
        for tmap_name in needed_tmaps_names:
            tmaps = update_tmaps(tmap_name=tmap_name, tmaps=tmaps)

        # Update args with TMaps
        args.tensor_maps_in = [tmaps[tmap_name] for tmap_name in args.input_tensors]
        args.tensor_maps_out = [tmaps[tmap_name] for tmap_name in args.output_tensors]
    if "bottleneck_type" in args:
        args.bottleneck_type = BOTTLENECK_STR_TO_ENUM[args.bottleneck_type]
    if "learning_rate_schedule" in args:
        if args.learning_rate_schedule is not None and args.patience < args.epochs:
            raise ValueError(
                "learning_rate_schedule is not compatible with ReduceLROnPlateau. "
                "Set patience > epochs.",
            )

    np.random.seed(args.random_seed)

    # Replace tildes with full path to home dirs
    if "bad_xml_dir" in args and args.bad_xml_dir:
        args.bad_xml_dir = os.path.expanduser(args.bad_xml_dir)
    if "bad_hd5_dir" in args and args.bad_hd5_dir:
        args.bad_hd5_dir = os.path.expanduser(args.bad_hd5_dir)

    logging.info(f"Command Line was: {command_line}")
    if "input_tensors" in args and args.recipe != "explore_icu":
        logging.info(f"Total TensorMaps: {len(tmaps)} Arguments are {args}")

    if args.num_workers <= 0:
        raise ValueError(
            f"num_workers must be a positive integer, got: {args.num_workers}",
        )

    if "debug" in args and args.debug:
        args.num_workers = 1

    if args.remap_layer is not None:
        args.remap_layer = {
            pretrained_layer: new_layer
            for pretrained_layer, new_layer in args.remap_layer
        }

    # Imports: third party
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
