# pylint: disable=wrong-import-order, wrong-import-position
# Imports: standard library
import os
import re
import math
import hashlib
import logging
import argparse
from typing import Dict, List, Tuple, Union, Callable, Optional
from datetime import datetime
from collections import OrderedDict

# Imports: third party
import numpy as np
import pydot
import pandas as pd
import seaborn as sns
import pygraphviz as pgv
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    roc_curve,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

# Imports: first party
from ml4c3.metrics import coefficient_of_determination
from ml4c3.datasets import (
    BATCH_IDS_INDEX,
    BATCH_INPUT_INDEX,
    make_dataset,
    tensors_to_sources,
    get_train_valid_test_ids,
)
from definitions.ecg import ECG_DATE_FORMAT, ECG_DATETIME_FORMAT
from definitions.models import BottleneckType
from tensormap.TensorMap import TensorMap, update_tmaps, find_negative_label_and_channel

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib      # isort:skip
matplotlib.use("Agg")  # isort:skip
from matplotlib import pyplot as plt  # isort:skip
# fmt: on


RECALL_LABEL = "Sensitivity | True Positive Rate | TP/(TP+FN)"
FALLOUT_LABEL = "1 - Specificity | False Positive Rate | FP/(FP+TN)"
PRECISION_LABEL = "Precision | Positive Predictive Value | TP/(TP+FP)"

SUBPLOT_SIZE = 8

COLOR_ARRAY = [
    "tan",
    "indigo",
    "cyan",
    "pink",
    "purple",
    "blue",
    "chartreuse",
    "deepskyblue",
    "green",
    "salmon",
    "aqua",
    "magenta",
    "aquamarine",
    "red",
    "coral",
    "tomato",
    "grey",
    "black",
    "maroon",
    "hotpink",
    "steelblue",
    "orange",
    "papayawhip",
    "wheat",
    "chocolate",
    "darkkhaki",
    "gold",
    "orange",
    "crimson",
    "slategray",
    "violet",
    "cadetblue",
    "midnightblue",
    "darkorchid",
    "paleturquoise",
    "plum",
    "lime",
    "teal",
    "peru",
    "silver",
    "darkgreen",
    "rosybrown",
    "firebrick",
    "saddlebrown",
    "dodgerblue",
    "orangered",
]

ECG_REST_PLOT_DEFAULT_YRANGE = 3.0
ECG_REST_PLOT_MAX_YRANGE = 10.0
ECG_REST_PLOT_LEADS = [
    ["strip_I", "strip_aVR", "strip_V1", "strip_V4"],
    ["strip_II", "strip_aVL", "strip_V2", "strip_V5"],
    ["strip_III", "strip_aVF", "strip_V3", "strip_V6"],
]
ECG_REST_PLOT_MEDIAN_LEADS = [
    ["median_I", "median_aVR", "median_V1", "median_V4"],
    ["median_II", "median_aVL", "median_V2", "median_V5"],
    ["median_III", "median_aVF", "median_V3", "median_V6"],
]
ECG_REST_PLOT_AMP_LEADS = [
    [0, 3, 6, 9],
    [1, 4, 7, 10],
    [2, 5, 8, 11],
]


def evaluate_predictions(
    tm: TensorMap,
    y_predictions: np.ndarray,
    y_truth: np.ndarray,
    title: str,
    image_ext: str,
    folder: str,
    test_paths: Optional[List[str]] = None,
    max_melt: int = 30000,
    rocs: List[Tuple[np.ndarray, np.ndarray, Dict[str, int]]] = [],
    scatters: List[Tuple[np.ndarray, np.ndarray, str, List[str]]] = [],
    data_split: str = "test",
) -> Dict[str, float]:
    """Evaluate predictions for a given TensorMap with truth data and plot the
    appropriate metrics. Accumulates data in the rocs and scatters lists to
    facilitate subplotting.

    :param tm: The TensorMap predictions to evaluate
    :param y_predictions: The predictions
    :param y_truth: The truth
    :param title: A title for the plots
    :param image_ext: File type to save images as
    :param folder: The folder to save the plots at
    :param test_paths: The tensor paths that were predicted
    :param max_melt: For multi-dimensional prediction the maximum number of
                     prediction to allow in the flattened array
    :param rocs: (output) List of Tuples which are inputs for ROC curve plotting to
                 allow subplotting downstream
    :param scatters: (output) List of Tuples which are inputs for scatter plots to
                     allow subplotting downstream
    :param data_split: The data split being evaluated (train, valid, or test)
    :return: Dictionary of performance metrics with string keys for labels and float
             values
    """
    performance_metrics = {}
    if tm.is_categorical and tm.axes == 1:
        logging.info(
            f"{data_split} split: {tm.name} has channel map: {tm.channel_map}"
            f" with {y_predictions.shape[0]} examples.\n"
            f"Sum Truth:{np.sum(y_truth, axis=0)} \nSum pred"
            f" :{np.sum(y_predictions, axis=0)}",
        )
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        plot_precision_recall_per_class(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            image_ext=image_ext,
            prefix=folder,
            data_split=data_split,
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            image_ext=image_ext,
            prefix=folder,
            data_split=data_split,
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
        # only plot confusion matrix for non-binary tasks
        if len(tm.channel_map) > 2:
            plot_confusion_matrix(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            )
    elif tm.is_categorical and tm.axes == 2:
        melt_shape = (
            y_predictions.shape[0] * y_predictions.shape[1],
            y_predictions.shape[2],
        )
        idx = np.random.choice(
            np.arange(melt_shape[0]),
            min(melt_shape[0], max_melt),
            replace=False,
        )
        y_predictions = y_predictions.reshape(melt_shape)[idx]
        y_truth = y_truth.reshape(melt_shape)[idx]
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            image_ext=image_ext,
            prefix=folder,
            data_split=data_split,
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_categorical and tm.axes == 3:
        melt_shape = (
            y_predictions.shape[0] * y_predictions.shape[1] * y_predictions.shape[2],
            y_predictions.shape[3],
        )
        idx = np.random.choice(
            np.arange(melt_shape[0]),
            min(melt_shape[0], max_melt),
            replace=False,
        )
        y_predictions = y_predictions.reshape(melt_shape)[idx]
        y_truth = y_truth.reshape(melt_shape)[idx]
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            image_ext=image_ext,
            prefix=folder,
            data_split=data_split,
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_categorical and tm.axes == 4:
        melt_shape = (
            y_predictions.shape[0]
            * y_predictions.shape[1]
            * y_predictions.shape[2]
            * y_predictions.shape[3],
            y_predictions.shape[4],
        )
        idx = np.random.choice(
            np.arange(melt_shape[0]),
            min(melt_shape[0], max_melt),
            replace=False,
        )
        y_predictions = y_predictions.reshape(melt_shape)[idx]
        y_truth = y_truth.reshape(melt_shape)[idx]
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        plot_prediction_calibration(
            prediction=y_predictions,
            truth=y_truth,
            labels=tm.channel_map,
            title=title,
            image_ext=image_ext,
            prefix=folder,
            data_split=data_split,
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_language:
        performance_metrics.update(
            plot_roc_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        performance_metrics.update(
            plot_precision_recall_per_class(
                prediction=y_predictions,
                truth=y_truth,
                labels=tm.channel_map,
                title=title,
                image_ext=image_ext,
                prefix=folder,
                data_split=data_split,
            ),
        )
        rocs.append((y_predictions, y_truth, tm.channel_map))
    elif tm.is_continuous:
        performance_metrics.update(
            plot_scatter(
                prediction=tm.rescale(y_predictions),
                truth=tm.rescale(y_truth),
                title=title,
                image_ext=image_ext,
                prefix=folder,
                paths=test_paths,
                data_split=data_split,
            ),
        )
        scatters.append(
            (tm.rescale(y_predictions), tm.rescale(y_truth), title, test_paths),
        )
    else:
        logging.warning(f"No evaluation clause for tensor map {tm.name}")

    return performance_metrics


def plot_metric_history(
    history,
    image_ext: str,
    prefix: str,
):
    plt.rcParams["font.size"] = 14
    row = 0
    col = 0
    total_plots = int(
        len(history.history) / 2,
    )  # divide by 2 because we plot validation and train histories together
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    _, axes = plt.subplots(
        rows,
        cols,
        figsize=(int(cols * SUBPLOT_SIZE), int(rows * SUBPLOT_SIZE)),
    )
    for k in sorted(history.history.keys()):
        if not k.startswith("val_") and not k.startswith("no_"):
            if isinstance(history.history[k][0], LearningRateSchedule):
                if training_steps is None:
                    raise NotImplementedError(
                        "cannot plot learning rate schedule without training_steps",
                    )
                history.history[k] = [
                    history.history[k][0](i * training_steps)
                    for i in range(len(history.history[k]))
                ]
            axes[row, col].plot(
                list(range(1, len(history.history[k]) + 1)),
                history.history[k],
            )
            k_split = str(k).replace("output_", "").split("_")
            k_title = " ".join(OrderedDict.fromkeys(k_split))
            axes[row, col].set_title(k_title)
            axes[row, col].set_xlabel("epoch")
            if "val_" + k in history.history:
                axes[row, col].plot(
                    list(range(1, len(history.history["val_" + k]) + 1)),
                    history.history["val_" + k],
                )
                labels = ["train", "valid"]
            else:
                labels = [k]
            axes[row, col].legend(labels, loc="upper left")

            row += 1
            if row == rows:
                row = 0
                col += 1
                if col >= cols:
                    break

    plt.tight_layout()
    title = "metric_history"
    figure_path = os.path.join(prefix, title + image_ext)
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved learning curves at: {figure_path}")


def plot_prediction_calibration(
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: Dict[str, int],
    title: str,
    image_ext: str,
    prefix: str = "./figures/",
    n_bins: int = 10,
    data_split: str = "test",
):
    """Plot calibration performance and compute Brier Score.

    :param prediction: Array of probabilistic predictions with shape
                       (num_samples, num_classes)
    :param truth: The true classifications of each class, one hot encoded of shape
                  (num_samples, num_classes)
    :param labels: Dictionary mapping strings describing each class to their
                   corresponding index in the arrays
    :param title: The name of this plot
    :param prefix: Optional path prefix where the plot will be saved
    :param n_bins: Number of bins to quantize predictions into
    :param data_split: The data split being plotted (train, valid, or test)
    """

    plt.rcParams["font.size"] = 14
    _, (ax1, ax3, ax2) = plt.subplots(3, figsize=(SUBPLOT_SIZE, 2 * SUBPLOT_SIZE))

    true_sums = np.sum(truth, axis=0)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated Brier score: 0.0")
    ax3.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated Brier score: 0.0")

    if len(labels) == 2:
        _, negative_label_idx = find_negative_label_and_channel(labels)

    for label, idx in labels.items():
        if len(labels) == 2 and idx == negative_label_idx:
            continue

        y_true = truth[..., labels[label]]
        y_prob = prediction[..., labels[label]]
        color = _hash_string_to_color(label)
        brier_score = brier_score_loss(
            y_true,
            prediction[..., labels[label]],
            pos_label=1,
        )
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true,
            y_prob,
            n_bins=n_bins,
        )
        ax3.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"{label} Brier score: {brier_score:0.3f}",
            color=color,
        )
        ax2.hist(
            y_prob,
            range=(0, 1),
            bins=n_bins,
            label=f"{label} n={true_sums[labels[label]]:.0f}",
            histtype="step",
            lw=2,
            color=color,
        )

        bins = stats.mstats.mquantiles(y_prob, np.arange(0.0, 1.0, 1.0 / n_bins))
        binids = np.digitize(y_prob, bins) - 1

        bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
        bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
        bin_total = np.bincount(binids, minlength=len(bins))

        nonzero = bin_total != 0
        prob_true = bin_true[nonzero] / bin_total[nonzero]
        prob_pred = bin_sums[nonzero] / bin_total[nonzero]
        ax1.plot(
            prob_pred,
            prob_true,
            "s-",
            label=f"{label} Brier score: {brier_score:0.3f}",
            color=color,
        )
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title(f'{title.replace("_", " ")}\nCalibration plot (equally sized bins)')
    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    ax3.set_title("Calibration plot (equally spaced bins)")
    plt.tight_layout()

    figure_path = os.path.join(
        prefix,
        "calibrations_" + title + "_" + data_split + image_ext,
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    logging.info(f"{data_split} split: saved calibration plot at: {figure_path}")
    plt.close()


def plot_confusion_matrix(
    prediction: np.ndarray,
    truth: np.ndarray,
    labels: Dict[str, int],
    title: str,
    image_ext: str,
    prefix: str,
    data_split: str,
):
    plt.rcParams["font.size"] = 14
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(SUBPLOT_SIZE, 2 * SUBPLOT_SIZE))
    idx_to_label = {idx: label for label, idx in labels.items()}
    labels = list(labels.keys())

    flatten_categorical = lambda categorical: np.array(
        [idx_to_label[category] for category in categorical.argmax(axis=-1)],
    )
    class_prediction = flatten_categorical(prediction)
    class_truth = flatten_categorical(truth)

    cms = []
    for matrix_title, normalize, ax in [
        (f"{title} confusion matrix, n = {len(class_truth)}", None, ax1),
        ("normalized to true classes", "true", ax2),
    ]:
        cm = confusion_matrix(
            y_true=class_truth,
            y_pred=class_prediction,
            labels=labels,
            normalize=normalize,
        )
        cms.append(cm)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        cm_display.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation="vertical")
        ax.set_title(matrix_title)

        cm_df = pd.DataFrame(
            cm,
            columns=[f"pred:{label}" for label in labels],
            index=[f"true:{label}" for label in labels],
        )
        with pd.option_context("display.float_format", "{:0.2f}".format):
            logging.info(f"{matrix_title}:\n{cm_df}")

    figure_path = os.path.join(
        prefix,
        f"confusion_matrix_{title}_{data_split}{image_ext}",
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.tight_layout()
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
    logging.info(f"{data_split} split: saved confusion matrix at: {figure_path}")
    return cms


def plot_scatter(
    prediction: np.array,
    truth: np.array,
    title: str,
    image_ext: str,
    data_split: str,
    prefix: str = "./figures/",
    paths=None,
    top_k=3,
    alpha=0.5,
):
    prediction = prediction.ravel()
    truth = truth.ravel()
    margin = float((np.max(truth) - np.min(truth)) / 100)
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(SUBPLOT_SIZE, 2 * SUBPLOT_SIZE))
    ax1.plot(
        [np.min(truth), np.max(truth)],
        [np.min(truth), np.max(truth)],
        linewidth=2,
    )
    ax1.plot(
        [np.min(prediction), np.max(prediction)],
        [np.min(prediction), np.max(prediction)],
        linewidth=4,
    )
    pearson = np.corrcoef(prediction.flatten(), truth.flatten())[
        1,
        0,
    ]  # corrcoef returns full covariance matrix
    big_r_squared = coefficient_of_determination(truth, prediction)
    logging.info(
        f"Pearson:{pearson:0.3f} r^2:{pearson*pearson:0.3f} R^2:{big_r_squared:0.3f}",
    )
    ax1.scatter(
        prediction,
        truth,
        label=(
            f"Pearson:{pearson:0.3f} r^2:{pearson*pearson:0.3f}"
            f" R^2:{big_r_squared:0.3f}"
        ),
        marker=".",
        alpha=alpha,
    )
    if paths is not None:
        diff = np.abs(prediction - truth)
        arg_sorted = diff.argsort()
        # The path of the best prediction, ie the inlier
        _text_on_plot(
            ax1,
            prediction[arg_sorted[0]] + margin,
            truth[arg_sorted[0]] + margin,
            os.path.basename(paths[arg_sorted[0]]),
        )
        # Plot the paths of the worst predictions ie the outliers
        for idx in arg_sorted[-top_k:]:
            _text_on_plot(
                ax1,
                prediction[idx] + margin,
                truth[idx] + margin,
                os.path.basename(paths[idx]),
            )

    ax1.set_xlabel("Predictions")
    ax1.set_ylabel("Actual")
    ax1.set_title(title + "\n")
    ax1.legend(loc="lower right")

    sns.kdeplot(prediction, label="Predicted", color="r", ax=ax2)
    sns.histplot(prediction, label="Predicted", color="r", ax=ax2, stat="density")
    sns.kdeplot(truth, label="Truth", color="b", ax=ax2)
    sns.histplot(truth, label="Truth", color="b", ax=ax2, stat="density")
    ax2.legend(loc="upper left")

    figure_path = os.path.join(prefix, f"scatter_{title}_{data_split}{image_ext}")
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    logging.info(f"Try to save scatter plot at: {figure_path}")
    plt.savefig(figure_path)
    plt.close()
    return {title + "_pearson": pearson}


def subplot_scatters(
    scatters: List[Tuple[np.ndarray, np.ndarray, str, List[str]]],
    data_split: str,
    image_ext: str,
    plot_path: str = "./figures/",
    top_k: int = 3,
    alpha: float = 0.5,
):
    row = 0
    col = 0
    total_plots = len(scatters)
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    _, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * SUBPLOT_SIZE, rows * SUBPLOT_SIZE),
    )
    for prediction, truth, title, paths in scatters:
        prediction = prediction.ravel()
        truth = truth.ravel()
        axes[row, col].plot(
            [np.min(truth), np.max(truth)],
            [np.min(truth), np.max(truth)],
        )
        axes[row, col].plot(
            [np.min(prediction), np.max(prediction)],
            [np.min(prediction), np.max(prediction)],
        )
        axes[row, col].scatter(prediction, truth, marker=".", alpha=alpha)
        margin = float((np.max(truth) - np.min(truth)) / 100)

        # If tensor paths are provided, plot file names of top_k outliers and #1 inlier
        if paths is not None:
            diff = np.abs(prediction - truth)
            arg_sorted = diff.argsort()
            # The path of the best prediction, ie the inlier
            _text_on_plot(
                axes[row, col],
                prediction[arg_sorted[0]] + margin,
                truth[arg_sorted[0]] + margin,
                os.path.basename(paths[arg_sorted[0]]),
            )
            # Plot the paths of the worst predictions ie the outliers
            for idx in arg_sorted[-top_k:]:
                _text_on_plot(
                    axes[row, col],
                    prediction[idx] + margin,
                    truth[idx] + margin,
                    os.path.basename(paths[idx]),
                )
        axes[row, col].set_xlabel("Predictions")
        axes[row, col].set_ylabel("Actual")
        axes[row, col].set_title(title + "\n")
        pearson = np.corrcoef(prediction.flatten(), truth.flatten())[1, 0]
        r2 = pearson * pearson
        big_r2 = coefficient_of_determination(truth.flatten(), prediction.flatten())
        axes[row, col].text(
            0,
            1,
            f"Pearson:{pearson:0.3f} r^2:{r2:0.3f} R^2:{big_r2:0.3f}",
            verticalalignment="bottom",
            transform=axes[row, col].transAxes,
        )

        row += 1
        if row == rows:
            row = 0
            col += 1
            if col >= cols:
                break

    figure_path = os.path.join(plot_path, f"scatters_together_{data_split}{image_ext}")
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.close()
    logging.info(f"{data_split} split: saved scatters together at: {figure_path}")


def _plot_ecg_text(
    data: Dict[str, Union[np.ndarray, str, Dict]],
    fig: plt.Figure,
    w: float,
    h: float,
) -> None:
    # top text
    dt = datetime.strptime(data["datetime"], ECG_DATETIME_FORMAT)
    dob = data["dob"]
    if dob != "":
        dob = datetime.strptime(dob, ECG_DATE_FORMAT)
        dob = f"{dob:%d-%b-%Y}".upper()
    age = -1
    if not np.isnan(data["age"]):
        age = int(data["age"])

    fig.text(
        0.17 / w,
        8.04 / h,
        f"{data['lastname']}, {data['firstname']}",
        weight="bold",
    )
    fig.text(3.05 / w, 8.04 / h, f"ID:{data['patientid']}", weight="bold")
    fig.text(4.56 / w, 8.04 / h, f"{dt:%d-%b-%Y %H:%M:%S}".upper(), weight="bold")
    fig.text(6.05 / w, 8.04 / h, f"{data['sitename']}", weight="bold")

    fig.text(0.17 / w, 7.77 / h, f"{dob} ({age} yr)", weight="bold")
    fig.text(0.17 / w, 7.63 / h, f"{data['sex']}".title(), weight="bold")
    fig.text(0.17 / w, 7.35 / h, "Room: ", weight="bold")
    fig.text(0.17 / w, 7.21 / h, f"Loc: {data['location']}", weight="bold")

    fig.text(2.15 / w, 7.77 / h, "Vent. rate", weight="bold")
    fig.text(2.15 / w, 7.63 / h, "PR interval", weight="bold")
    fig.text(2.15 / w, 7.49 / h, "QRS duration", weight="bold")
    fig.text(2.15 / w, 7.35 / h, "QT/QTc", weight="bold")
    fig.text(2.15 / w, 7.21 / h, "P-R-T axes", weight="bold")

    fig.text(3.91 / w, 7.77 / h, f"{int(data['rate_md'])}", weight="bold", ha="right")
    fig.text(3.91 / w, 7.63 / h, f"{int(data['pr_md'])}", weight="bold", ha="right")
    fig.text(3.91 / w, 7.49 / h, f"{int(data['qrs_md'])}", weight="bold", ha="right")
    fig.text(
        3.91 / w,
        7.35 / h,
        f"{int(data['qt_md'])}/{int(data['qtc_md'])}",
        weight="bold",
        ha="right",
    )
    fig.text(
        3.91 / w,
        7.21 / h,
        f"{int(data['paxis_md'])}   {int(data['raxis_md'])}",
        weight="bold",
        ha="right",
    )

    fig.text(4.30 / w, 7.77 / h, "BPM", weight="bold", ha="right")
    fig.text(4.30 / w, 7.63 / h, "ms", weight="bold", ha="right")
    fig.text(4.30 / w, 7.49 / h, "ms", weight="bold", ha="right")
    fig.text(4.30 / w, 7.35 / h, "ms", weight="bold", ha="right")
    fig.text(4.30 / w, 7.21 / h, f"{int(data['taxis_md'])}", weight="bold", ha="right")

    fig.text(4.75 / w, 7.21 / h, f"{data['read_md']}", wrap=True, weight="bold")

    fig.text(1.28 / w, 6.65 / h, f"Technician: {''}", weight="bold")
    fig.text(1.28 / w, 6.51 / h, f"Test ind: {''}", weight="bold")
    fig.text(4.75 / w, 6.25 / h, f"Referred by: {''}", weight="bold")
    fig.text(7.63 / w, 6.25 / h, f"Electronically Signed By: {''}", weight="bold")


def _plot_ecg_full(voltage: Dict[str, np.ndarray], ax: plt.Axes) -> None:
    full_voltage = np.full((12, 2500), np.nan)
    for i, lead in enumerate(voltage):
        full_voltage[i] = voltage[lead]

    # convert voltage to millivolts
    full_voltage /= 1000

    # calculate space between leads
    min_y, max_y = ax.get_ylim()
    y_offset = (max_y - min_y) / len(voltage)

    text_xoffset = 5
    text_yoffset = -0.01

    # plot signal and add labels
    for i, lead in enumerate(voltage):
        this_offset = (len(voltage) - i - 0.5) * y_offset
        ax.plot(full_voltage[i] + this_offset, color="black", linewidth=0.375)
        ax.text(
            0 + text_xoffset,
            this_offset + text_yoffset,
            lead,
            ha="left",
            va="top",
            weight="bold",
        )


def _plot_ecg_clinical(voltage: Dict[str, np.ndarray], ax: plt.Axes) -> None:
    # get voltage in clinical chunks
    clinical_voltage = np.full((6, 2500), np.nan)
    halfgap = 5

    clinical_voltage[0][0 : 625 - halfgap] = voltage["I"][0 : 625 - halfgap]
    clinical_voltage[0][625 + halfgap : 1250 - halfgap] = voltage["aVR"][
        625 + halfgap : 1250 - halfgap
    ]
    clinical_voltage[0][1250 + halfgap : 1875 - halfgap] = voltage["V1"][
        1250 + halfgap : 1875 - halfgap
    ]
    clinical_voltage[0][1875 + halfgap : 2500] = voltage["V4"][1875 + halfgap : 2500]

    clinical_voltage[1][0 : 625 - halfgap] = voltage["II"][0 : 625 - halfgap]
    clinical_voltage[1][625 + halfgap : 1250 - halfgap] = voltage["aVL"][
        625 + halfgap : 1250 - halfgap
    ]
    clinical_voltage[1][1250 + halfgap : 1875 - halfgap] = voltage["V2"][
        1250 + halfgap : 1875 - halfgap
    ]
    clinical_voltage[1][1875 + halfgap : 2500] = voltage["V5"][1875 + halfgap : 2500]

    clinical_voltage[2][0 : 625 - halfgap] = voltage["III"][0 : 625 - halfgap]
    clinical_voltage[2][625 + halfgap : 1250 - halfgap] = voltage["aVF"][
        625 + halfgap : 1250 - halfgap
    ]
    clinical_voltage[2][1250 + halfgap : 1875 - halfgap] = voltage["V3"][
        1250 + halfgap : 1875 - halfgap
    ]
    clinical_voltage[2][1875 + halfgap : 2500] = voltage["V6"][1875 + halfgap : 2500]

    clinical_voltage[3] = voltage["V1"]
    clinical_voltage[4] = voltage["II"]
    clinical_voltage[5] = voltage["V5"]

    voltage = clinical_voltage

    # convert voltage to millivolts
    voltage /= 1000

    # calculate space between leads
    min_y, max_y = ax.get_ylim()
    y_offset = (max_y - min_y) / len(voltage)

    text_xoffset = 5
    text_yoffset = -0.1

    # plot signal and add labels
    for i, _ in enumerate(voltage):
        this_offset = (len(voltage) - i - 0.5) * y_offset
        ax.plot(voltage[i] + this_offset, color="black", linewidth=0.375)
        if i == 0:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "I",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                625 + text_xoffset,
                this_offset + text_yoffset,
                "aVR",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1250 + text_xoffset,
                this_offset + text_yoffset,
                "V1",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1875 + text_xoffset,
                this_offset + text_yoffset,
                "V4",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 1:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "II",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                625 + text_xoffset,
                this_offset + text_yoffset,
                "aVL",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1250 + text_xoffset,
                this_offset + text_yoffset,
                "V2",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1875 + text_xoffset,
                this_offset + text_yoffset,
                "V5",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 2:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "III",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                625 + text_xoffset,
                this_offset + text_yoffset,
                "aVF",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1250 + text_xoffset,
                this_offset + text_yoffset,
                "V3",
                ha="left",
                va="top",
                weight="bold",
            )
            ax.text(
                1875 + text_xoffset,
                this_offset + text_yoffset,
                "V6",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 3:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "V1",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 4:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "II",
                ha="left",
                va="top",
                weight="bold",
            )
        elif i == 5:
            ax.text(
                0 + text_xoffset,
                this_offset + text_yoffset,
                "V5",
                ha="left",
                va="top",
                weight="bold",
            )


def _plot_ecg_figure(
    patient_id: int,
    data: Dict[str, Union[np.ndarray, str, float]],
    plot_signal_function: Callable[[Dict[str, np.ndarray], plt.Axes], None],
    plot_mode: str,
    output_folder: str,
    image_ext: str,
) -> str:
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9.5

    w, h = 11, 8.5
    fig = plt.figure(figsize=(w, h), dpi=100)

    # patient info and ecg text
    _plot_ecg_text(data, fig, w, h)

    # define plot area in inches
    left = 0.17
    bottom = h - 7.85
    width = w - 2 * left
    height = h - bottom - 2.3

    # ecg plot area
    ax = fig.add_axes([left / w, bottom / h, width / w, height / h])

    # voltage is in microvolts
    # the entire plot area is 5.55 inches tall, 10.66 inches wide (141 mm, 271 mm)
    # the resolution on the y-axis is 10 mm/mV
    # the resolution on the x-axis is 25 mm/s
    inch2mm = lambda inches: inches * 25.4

    # 1. set y-limit to max 14.1 mV
    y_res = 10  # mm/mV
    max_y = inch2mm(height) / y_res
    min_y = 0
    ax.set_ylim(min_y, max_y)

    # 2. set x-limit to max 10.8 s, center 10 s leads
    sampling_frequency = 250  # Hz
    x_res = 25  # mm/s
    max_x = inch2mm(width) / x_res
    x_buffer = (max_x - 10) / 2
    max_x -= x_buffer
    min_x = -x_buffer
    max_x *= sampling_frequency
    min_x *= sampling_frequency
    ax.set_xlim(min_x, max_x)

    # 3. set ticks for every 0.1 mV or every 1/25 s
    y_tick = 1 / y_res
    x_tick = 1 / x_res * sampling_frequency
    x_major_ticks = np.arange(min_x, max_x, x_tick * 5)
    x_minor_ticks = np.arange(min_x, max_x, x_tick)
    y_major_ticks = np.arange(min_y, max_y, y_tick * 5)
    y_minor_ticks = np.arange(min_y, max_y, y_tick)

    ax.set_xticks(x_major_ticks)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_major_ticks)
    ax.set_yticks(y_minor_ticks, minor=True)

    ax.tick_params(
        which="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )
    ax.grid(b=True, color="r", which="major", lw=0.5)
    ax.grid(b=True, color="r", which="minor", lw=0.2)

    # signal plot
    voltage = data["2500"]
    plot_signal_function(voltage, ax)

    # bottom text
    fig.text(
        0.17 / w,
        0.46 / h,
        f"{x_res}mm/s    {y_res}mm/mV    {sampling_frequency}Hz",
        ha="left",
        va="center",
        weight="bold",
    )

    # save both pdf and image
    title = re.sub(r"[:/. ]", "", f'{patient_id}_{data["datetime"]}_{plot_mode}')
    fpath = os.path.join(output_folder, f"{title}{image_ext}")
    plt.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_ecg(args):
    plot_tensors = [
        "ecg_patientid",
        "ecg_firstname",
        "ecg_lastname",
        "ecg_sex",
        "ecg_dob",
        "ecg_age",
        "ecg_datetime",
        "ecg_sitename",
        "ecg_location",
        "ecg_read_md",
        "ecg_taxis_md",
        "ecg_rate_md",
        "ecg_pr_md",
        "ecg_qrs_md",
        "ecg_qt_md",
        "ecg_paxis_md",
        "ecg_raxis_md",
        "ecg_qtc_md",
    ]
    voltage_tensor = "12_lead_ecg_2500"
    needed_tensors = plot_tensors + [voltage_tensor]

    tmaps = {}
    for needed_tensor in needed_tensors:
        tmaps = update_tmaps(needed_tensor, tmaps)
    tensor_maps_in = [tmaps[it] for it in needed_tensors]

    if args.plot_mode == "clinical":

        plot_signal_function = _plot_ecg_clinical
    elif args.plot_mode == "full":
        plot_signal_function = _plot_ecg_full
    else:
        raise ValueError(f"Unsupported plot mode: {args.plot_mode}")

    patient_ids, _, _ = get_train_valid_test_ids(
        tensors=args.tensors,
        mrn_column_name=args.mrn_column_name,
        patient_csv=args.patient_csv,
        valid_ratio=0,
        test_ratio=0,
        allow_empty_split=True,
    )
    hd5_sources, _ = tensors_to_sources(args.tensors, tensor_maps_in)
    dataset, stats, cleanup = make_dataset(
        data_split="plot_ecg",
        hd5_sources=hd5_sources,
        csv_sources=[],
        patient_ids=patient_ids,
        input_tmaps=tensor_maps_in,
        output_tmaps=[],
        batch_size=1,
        num_workers=args.num_workers,
        cache_off=True,
        augment=False,
        validate=False,
        normalize=False,
        keep_ids=True,
        verbose=False,
    )

    dataset = dataset.unbatch()

    # Unbatched dataset returns tuples of dictionaries of tensors:
    # The elements of the tuples are the input and output data.
    # The elements of the dictionaries are tensor names and values.
    i = -1
    for i, ecg in enumerate(dataset):
        patient_id = ecg[BATCH_IDS_INDEX]
        ecg = ecg[BATCH_INPUT_INDEX]

        # Make dictionary keys friendlier, extract and process tensors
        data = {}
        for k, v in ecg.items():
            # "input_ecg_patientid_language" --> "patientid"
            new_k = k.split("ecg_", 1)[1].rsplit("_", 1)[0]

            # Extract voltage tensor into dictionary of numpy arrays, keyed by lead
            if new_k == "2500":
                new_v = {}
                for (
                    lead,
                    idx,
                ) in tmaps[voltage_tensor].channel_map.items():
                    new_v[lead] = ecg[k][:, idx].numpy()
            # Extract categorical variable as category label
            elif new_k == "sex":
                idx_to_label = {v: k for k, v in tmaps["ecg_sex"].channel_map.items()}
                new_v = idx_to_label[ecg[k].numpy().argmax()]
            # Extract all other tensors, cast to string if necessary
            else:
                new_v = ecg[k][0].numpy()
                if isinstance(new_v, bytes):
                    new_v = new_v.decode()
            data[new_k] = new_v

        fpath = _plot_ecg_figure(
            patient_id=patient_id,
            data=data,
            plot_signal_function=plot_signal_function,
            plot_mode=args.plot_mode,
            output_folder=args.output_folder,
            image_ext=args.image_ext,
        )
        logging.info(f"Saved ECG {i + 1} to {fpath}")
    logging.info(f"Saved {i + 1} ECGs to {args.output_folder}")

    cleanup()


def get_fpr_tpr_roc_pred(y_pred, test_truth, labels):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for k in labels:
        cur_idx = labels[k]
        if y_pred.shape[1] == 1:
            y_pred_this_label = y_pred[:, 0]
        elif y_pred.shape[1] > 1:
            if len(labels) > 1:
                y_pred_this_label = y_pred[
                    :, cur_idx if len(labels) > 1 else cur_idx + 1
                ]
            else:
                y_pred_this_label = y_pred[:, cur_idx + 1]
        fpr[cur_idx], tpr[cur_idx], _ = roc_curve(
            test_truth[:, cur_idx],
            y_pred_this_label,
        )
        roc_auc[cur_idx] = auc(fpr[cur_idx], tpr[cur_idx])
    return fpr, tpr, roc_auc


def plot_roc_per_class(
    prediction: np.array,
    truth: np.array,
    labels: dict,
    title: str,
    image_ext: str,
    prefix: str = "./figures/",
    data_split: str = "test",
):
    plt.rcParams["font.size"] = 14
    lw = 2
    labels_to_areas = {}
    true_sums = np.sum(truth, axis=0)
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))
    fpr, tpr, roc_auc = get_fpr_tpr_roc_pred(prediction, truth, labels)

    if len(labels) == 2:
        _, negative_label_idx = find_negative_label_and_channel(labels)

    for label, idx in labels.items():
        if len(labels) == 2 and idx == negative_label_idx:
            continue

        labels_to_areas[label] = roc_auc[labels[label]]
        color = _hash_string_to_color(label)
        label_text = (
            f"{label} = {roc_auc[labels[label]]:.3f}, n={true_sums[labels[label]]:.0f}"
        )
        plt.plot(
            fpr[labels[label]],
            tpr[labels[label]],
            color=color,
            lw=lw,
            label=label_text,
        )
        logging.info(
            f"{data_split} split: ROC AUC for {label_text}, "
            f"Truth shape {truth.shape}, "
            f"True sums {true_sums}",
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([-0.02, 1.03])
    plt.ylabel(RECALL_LABEL)
    plt.xlabel(FALLOUT_LABEL)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "k:", lw=0.5)
    plt.title(f"ROC curve: {title}, n={truth.shape[0]:.0f}\n")

    figure_path = os.path.join(
        prefix,
        "per_class_roc_" + title + "_" + data_split + image_ext,
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
    logging.info(f"{data_split} split: saved ROC curve at: {figure_path}")
    return labels_to_areas


def subplot_rocs(
    rocs: List[Tuple[np.ndarray, np.ndarray, Dict[str, int]]],
    data_split: str,
    image_ext: str,
    plot_path: str = "./figures/",
):
    """
    Log and tabulate AUCs given as nested dictionaries in the format
    '{model: {label: auc}}'
    """
    lw = 2
    row = 0
    col = 0
    total_plots = len(rocs)
    cols = max(2, int(math.ceil(math.sqrt(total_plots))))
    rows = max(2, int(math.ceil(total_plots / cols)))
    _, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * SUBPLOT_SIZE, rows * SUBPLOT_SIZE),
    )
    for predicted, truth, labels in rocs:
        true_sums = np.sum(truth, axis=0)
        fpr, tpr, roc_auc = get_fpr_tpr_roc_pred(predicted, truth, labels)

        if len(labels) == 2:
            _, negative_label_idx = find_negative_label_and_channel(labels)

        for label, idx in labels.items():
            if len(labels) == 2 and idx == negative_label_idx:
                continue

            color = _hash_string_to_color(label)
            label_text = (
                f"{label} area: {roc_auc[labels[label]]:.3f}"
                f" n={true_sums[labels[label]]:.0f}"
            )
            axes[row, col].plot(
                fpr[labels[label]],
                tpr[labels[label]],
                color=color,
                lw=lw,
                label=label_text,
            )
            logging.info(f"ROC Label {label_text}")
        axes[row, col].set_xlim([0.0, 1.0])
        axes[row, col].set_ylim([-0.02, 1.03])
        axes[row, col].set_ylabel(RECALL_LABEL)
        axes[row, col].set_xlabel(FALLOUT_LABEL)
        axes[row, col].legend(loc="lower right")
        axes[row, col].plot([0, 1], [0, 1], "k:", lw=0.5)
        axes[row, col].set_title(f"ROC n={np.sum(true_sums):.0f}")

        row += 1
        if row == rows:
            row = 0
            col += 1
            if col >= cols:
                break
    figure_path = os.path.join(plot_path, f"roc-together-{data_split}{image_ext}")
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path)
    plt.close()


def plot_precision_recall_per_class(
    prediction: np.array,
    truth: np.array,
    labels: dict,
    title: str,
    image_ext: str,
    prefix: str = "./figures/",
    data_split: str = "test",
):
    plt.rcParams["font.size"] = 14
    lw = 2.0
    labels_to_areas = {}
    true_sums = np.sum(truth, axis=0)
    plt.figure(figsize=(SUBPLOT_SIZE, SUBPLOT_SIZE))

    if len(labels) == 2:
        _, negative_label_idx = find_negative_label_and_channel(labels)

    for label, idx in labels.items():
        if len(labels) == 2 and idx == negative_label_idx:
            continue

        precision, recall, _ = precision_recall_curve(
            truth[:, labels[label]],
            prediction[:, labels[label]],
        )
        average_precision = average_precision_score(
            truth[:, labels[label]],
            prediction[:, labels[label]],
        )
        labels_to_areas[label] = average_precision
        color = _hash_string_to_color(label)
        label_text = (
            f"{label} mean precision: {average_precision:.3f},"
            f" n={true_sums[labels[label]]:.0f}"
        )
        plt.plot(recall, precision, lw=lw, color=color, label=label_text)
        logging.info(f"{data_split} split: prAUC {label_text}")

    plt.xlim([0.0, 1.0])
    plt.ylim([-0.02, 1.03])
    plt.xlabel(RECALL_LABEL)
    plt.ylabel(PRECISION_LABEL)
    plt.legend(loc="lower right")
    plt.title(f"PR curve: {title}, n={np.sum(true_sums):.0f}\n")

    figure_path = os.path.join(
        prefix,
        "precision-recall-" + title + "-" + data_split + image_ext,
    )
    if not os.path.exists(os.path.dirname(figure_path)):
        os.makedirs(os.path.dirname(figure_path))
    plt.savefig(figure_path, bbox_inches="tight")
    plt.close()
    logging.info(f"{data_split} split: saved precision-recall curve at: {figure_path}")
    return labels_to_areas


def _hash_string_to_color(string):
    """
    Hash a string to color (using hashlib and not the built-in hash for consistency
    between runs)
    """
    return COLOR_ARRAY[
        int(hashlib.sha1(string.encode("utf-8")).hexdigest(), 16) % len(COLOR_ARRAY)
    ]


def _text_on_plot(axes, x, y, text, alpha=0.8, background="white"):
    t = axes.text(x, y, text)
    t.set_bbox({"facecolor": background, "alpha": alpha, "edgecolor": background})


def plot_architecture_diagram(dot: pydot.Dot, image_path: str):
    """
    Given a graph representation of a model architecture,
    save the architecture diagram to an image file.

    :param dot: pydot.Dot representation of model
    :param image_path: path to save svg of architecture diagram to
    """
    legend = {}
    for n in dot.get_nodes():
        if n.get_label():
            if "Conv1" in n.get_label():
                legend["Conv1"] = "cyan"
                n.set_fillcolor("cyan")
            elif "Conv2" in n.get_label():
                legend["Conv2"] = "deepskyblue1"
                n.set_fillcolor("deepskyblue1")
            elif "Conv3" in n.get_label():
                legend["Conv3"] = "deepskyblue3"
                n.set_fillcolor("deepskyblue3")
            elif "UpSampling" in n.get_label():
                legend["UpSampling"] = "darkslategray2"
                n.set_fillcolor("darkslategray2")
            elif "Transpose" in n.get_label():
                legend["Transpose"] = "deepskyblue2"
                n.set_fillcolor("deepskyblue2")
            elif "BatchNormalization" in n.get_label():
                legend["BatchNormalization"] = "goldenrod1"
                n.set_fillcolor("goldenrod1")
            elif "output_" in n.get_label():
                n.set_fillcolor("darkolivegreen2")
                legend["Output"] = "darkolivegreen2"
            elif "softmax" in n.get_label():
                n.set_fillcolor("chartreuse")
                legend["softmax"] = "chartreuse"
            elif "MaxPooling" in n.get_label():
                legend["MaxPooling"] = "aquamarine"
                n.set_fillcolor("aquamarine")
            elif "Dense" in n.get_label():
                legend["Dense"] = "gold"
                n.set_fillcolor("gold")
            elif "Reshape" in n.get_label():
                legend["Reshape"] = "coral"
                n.set_fillcolor("coral")
            elif "Input" in n.get_label():
                legend["Input"] = "darkolivegreen1"
                n.set_fillcolor("darkolivegreen1")
            elif "Activation" in n.get_label():
                legend["Activation"] = "yellow"
                n.set_fillcolor("yellow")
        n.set_style("filled")

    for label in legend:
        legend_node = pydot.Node(
            "legend" + label,
            label=label,
            shape="box",
            fillcolor=legend[label],
        )
        dot.add_node(legend_node)

    if image_path.endswith("svg"):
        # pydot svg applies a scale factor that clips the image so unscale it
        svg_string = dot.create_svg().decode()
        svg_string = re.sub(
            r"scale\(\d+\.?\d+\ \d+\.?\d+\)",
            "scale(1 1)",
            svg_string,
        ).encode()
        with open(image_path, "wb") as f:
            f.write(svg_string)
    elif image_path.endswith("png"):
        dot.write_png(path=image_path)
    else:
        dot.write(
            path=image_path,
            prog="dot",
            format=os.path.splitext(image_path)[-1][1:],
        )

    logging.info(f"Saved architecture diagram to: {image_path}")


def _conv_layer(dimension: int, conv_type: str) -> str:
    if dimension == 4 and conv_type == "conv":
        conv_layer = "Conv3D"
    elif dimension == 3 and conv_type == "conv":
        conv_layer = "Conv2D"
    elif dimension == 2 and conv_type == "conv":
        conv_layer = "Conv1D"
    elif dimension == 3 and conv_type == "separable":
        conv_layer = "SeparableConv2D"
    elif dimension == 2 and conv_type == "separable":
        conv_layer = "SeparableConv1D"
    elif dimension == 3 and conv_type == "depth":
        conv_layer = "DepthwiseConv2D"
    else:
        raise ValueError(f"Unknown conv_type/dimension: {conv_type}/{dimension}")
    return conv_layer


def _activation_layer(activation_layer: str) -> str:
    if activation_layer.lower() == "relu":
        return "ReLU"
    return activation_layer


def _normalization_layer(normalization_layer: str) -> str:
    if normalization_layer.lower() == "batch_norm":
        return "BN"
    return normalization_layer


def _pool_layer(pool_type: str) -> str:
    if pool_type == "max":
        return "MaxPool"
    elif pool_type == "average":
        return "AveragePool"
    elif pool_type == "upsample":
        return "Upsample"
    return pool_type


def _conv_label(
    args: argparse.Namespace,
    layer_order: List[str],
    conv_layer: str,
) -> str:
    conv_label = (
        "<<table border='0' cellborder='1' cellspacing='0' cellpadding='10'><tr>"
    )
    for layer in layer_order:
        if layer == "convolution":
            conv_label += f"<td port='{layer}' bgcolor='cyan'>{conv_layer}</td>"
        elif layer == "activation":
            if args.activation_layer:
                conv_label += f"<td port='{layer}' bgcolor='yellow'>{_activation_layer(args.activation_layer)}</td>"
        elif layer == "normalization":
            if args.normalization_layer:
                conv_label += f"<td port='{layer}' bgcolor='orange'>{_normalization_layer(args.normalization_layer)}</td>"
        elif layer == "dropout":
            if args.spatial_dropout > 0:
                conv_label += f"<td port='{layer}' bgcolor='grey'>Spatial Dropout</td>"
    conv_label += "</tr></table>>"
    return conv_label


INVISIBLE_ARGS = {
    "label": "",
    "style": "invisible",
    "fixedsize": True,
    "width": 0,
    "height": 0,
}


def _conv_block(
    args: argparse.Namespace,
    input_idx: int,
    conv_label: str,
    g: pgv.AGraph,
    last_node: str,
) -> str:
    conv_block = []

    for i in range(args.conv_block_size):
        node = f"{input_idx}_conv_layer_{i}"
        g.add_node(node, label=conv_label, shape="none", margin=0)
        g.add_edge(last_node, node)
        last_node = node
        conv_block.append(node)

    if args.pool_type:
        node = f"{input_idx}_conv_pool"
        g.add_node(
            node,
            label=_pool_layer(args.pool_type),
            style="filled",
            fillcolor="aquamarine",
        )
        g.add_edge(last_node, node)
        last_node = node
        conv_block.append(node)

    g.add_subgraph(
        conv_block,
        name=f"cluster_{input_idx}_conv_block",
        style="dotted",
        label=f"Convolutional Block x{len(args.conv_blocks)}",
        labeljust="r",
        labelloc="b",
    )
    return last_node


def _res_block(
    args: argparse.Namespace,
    input_idx: int,
    conv_label: str,
    g: pgv.AGraph,
    last_node: str,
) -> str:
    res_block = []

    node = f"{input_idx}_res_start"
    start_node = node
    g.add_node(node, **INVISIBLE_ARGS)
    g.add_edge(last_node, node, arrowhead="none")
    last_node = node
    res_block.append(node)

    for i in range(args.residual_block_size):
        node = f"{input_idx}_res_layer_{i}"
        g.add_node(node, label=conv_label, shape="none", margin=0)
        g.add_edge(last_node, node)
        last_node = node
        res_block.append(node)

    node = f"{input_idx}_res_end"
    end_node = node
    g.add_node(node, label="Add", style="filled", fillcolor="aquamarine")
    g.add_edge(last_node, node)
    last_node = node
    res_block.append(node)

    g.add_edge(
        start_node,
        end_node,
        constraint=False,
        xlabel="match dimensions via point conv",
    )

    if args.pool_type:
        node = f"{input_idx}_res_pool"
        g.add_node(
            node,
            label=_pool_layer(args.pool_type),
            style="filled",
            fillcolor="aquamarine",
        )
        g.add_edge(last_node, node)
        last_node = node
        res_block.append(node)

    g.add_subgraph(
        res_block,
        name=f"cluster_{input_idx}_res_block",
        style="dotted",
        label=f"Residual Block x{len(args.residual_blocks)}",
        labeljust="r",
        labelloc="b",
    )
    return last_node


def _dense_block(
    args: argparse.Namespace,
    input_idx: int,
    conv_label: str,
    g: pgv.AGraph,
    last_node: str,
) -> str:
    dense_block = []

    node = f"{input_idx}_dense_start"
    g.add_node(node, **INVISIBLE_ARGS)
    g.add_edge(last_node, node, arrowhead="none")
    last_node = node
    last_concat = node
    dense_block.append(node)

    for i in range(args.dense_block_size):
        node = f"{input_idx}_dense_layer_{i}"
        g.add_node(node, label=conv_label, shape="none", margin=0)
        g.add_edge(last_node, node)
        last_node = node
        dense_block.append(node)

        if i < args.dense_block_size - 1:
            node = f"{input_idx}_dense_connect_{i}"
            g.add_node(node, label="Concat", style="filled", fillcolor="aquamarine")
            g.add_edge(last_node, node)
            last_node = node
            dense_block.append(node)
            g.add_edge(last_concat, node, constraint=False)
            last_concat = node

    if args.pool_type:
        node = f"{input_idx}_dense_pool"
        g.add_node(
            node,
            label=_pool_layer(args.pool_type),
            style="filled",
            fillcolor="aquamarine",
        )
        g.add_edge(last_node, node)
        last_node = node
        dense_block.append(node)

    g.add_subgraph(
        dense_block,
        name=f"cluster_{input_idx}_dense_block",
        style="dotted",
        label=f"Dense Block x{len(args.dense_blocks)}",
        labeljust="r",
        labelloc="b",
    )
    return last_node


def _fully_connected_label(args: argparse.Namespace, units: int) -> str:
    label = "<<table border='0' cellborder='1' cellspacing='0' cellpadding='10'><tr>"
    for layer in args.dense_layer_order:
        if layer == "dense":
            label += f"<td port='{layer}' bgcolor='deepskyblue'>Dense ({units})</td>"
        elif layer == "activation":
            if args.activation_layer:
                label += f"<td port='{layer}' bgcolor='yellow'>{_activation_layer(args.activation_layer)}</td>"
        elif layer == "normalization":
            if args.normalization_layer:
                label += f"<td port='{layer}' bgcolor='orange'>{_normalization_layer(args.normalization_layer)}</td>"
        elif layer == "dropout":
            if args.dense_dropout > 0:
                label += f"<td port='{layer}' bgcolor='grey'>Dropout</td>"
    label += "</tr></table>>"

    return label


def plot_condensed_architecture_diagram(args: argparse.Namespace) -> pgv.AGraph:
    g = pgv.AGraph(directed=True)
    g.graph_attr.update(rankdir="LR", splines="ortho", fontname="arial")
    g.node_attr.update(shape="box", fontname="arial")
    g.edge_attr.update(fontname="arial")

    # Inputs/Encoders
    encoders = []
    for input_idx, tm in enumerate(args.tensor_maps_in):
        node = f"{input_idx}_{tm.name}"
        g.add_node(node, label=tm.name, style="filled", fillcolor="green")
        last_node = node
        dimension = tm.axes
        if dimension > 1:
            # Input conv blocks
            conv_layer = _conv_layer(dimension, args.conv_type)

            if args.conv_blocks:
                conv_label = _conv_label(args, args.conv_block_layer_order, conv_layer)
                last_node = _conv_block(args, input_idx, conv_label, g, last_node)
            if args.residual_blocks:
                conv_label = _conv_label(
                    args,
                    args.residual_block_layer_order,
                    conv_layer,
                )
                last_node = _res_block(args, input_idx, conv_label, g, last_node)
            if args.dense_blocks:
                conv_label = _conv_label(args, args.dense_block_layer_order, conv_layer)
                last_node = _dense_block(args, input_idx, conv_label, g, last_node)

            node = f"{input_idx}_flatten"
            g.add_node(node, label="Flatten", style="filled", fillcolor="aquamarine")
            g.add_edge(last_node, node)
            last_node = node
        elif tm.annotation_units > 0:
            # Input fully connected block
            label = _fully_connected_label(args, tm.annotation_units)

            node = f"{input_idx}_fully_connected_input"
            g.add_node(node, label=label, shape="none", margin=0)
            g.add_edge(last_node, node)
            last_node = node
        else:
            # Direct input
            pass
        encoders.append(last_node)

    # Bottleneck
    if args.bottleneck_type != BottleneckType.FlattenRestructure:
        raise NotImplementedError(
            f"Simple model architecture diagram for bottleneck type "
            f"({args.bottleneck_type}) is not yet supported.",
        )

    if len(encoders) > 1:
        node = "bottleneck"
        g.add_node(node, label="Concat", style="filled", fillcolor="aquamarine")
        for encoder in encoders:
            g.add_edge(encoder, node)
        last_node = node

    # Fully connected layers
    for i, layer in enumerate(args.dense_layers):
        label = _fully_connected_label(args, layer)
        node = f"{i}_fully_connected"
        g.add_node(node, label=label, shape="none", margin=0)
        g.add_edge(last_node, node)
        last_node = node

    node = f"predecoder"
    g.add_node(node, **INVISIBLE_ARGS)
    g.add_edge(last_node, node, arrowhead="none")
    last_node = node

    # Outputs/Decoders
    for output_idx, tm in enumerate(args.tensor_maps_out):
        dimension = tm.axes
        if dimension > 1:
            raise NotImplementedError(
                f"Simple model architecture diagram for multidimensional outputs "
                f"({tm}) is not yet supported.",
            )
        else:
            node = f"{output_idx}_{tm.name}"
            g.add_node(node, label=tm.name, style="filled", fillcolor="green")
            g.add_edge(last_node, node)

    image_path = os.path.join(
        args.output_folder,
        f"condensed-architecture{args.image_ext}",
    )
    g.draw(image_path, prog="dot")
    logging.info(f"Saved architecture diagram to: {image_path}")

    return g


def plot_feature_coefficients(
    plot_path: str,
    model_name: str,
    feature_values: pd.DataFrame,
    top_features_to_plot: int,
    image_ext: str,
):
    sns.set(style="white", palette="muted", color_codes=True)
    sns.set_context("talk")

    # Isolate subset of top features for plotting
    if top_features_to_plot < len(feature_values):
        feature_values = feature_values.reindex(
            feature_values["coefficient"].abs().sort_values(ascending=False).index,
        )
        feature_values = feature_values.iloc[:top_features_to_plot]

    feature_names = feature_values["feature"].to_list()
    for k, feature_name in enumerate(feature_names):
        if len(feature_name) > 51:
            feature_names[k] = feature_name[:24] + "..." + feature_name[-24:]

    # Calculate length of strings to enable dynamic resizing of figure
    length_longest_feature_string = max([len(feature) for feature in feature_names])
    fig_width = 8 + length_longest_feature_string / 10
    fig_height = 2 + top_features_to_plot * 0.35

    plt.figure(
        num=None,
        figsize=(fig_width, fig_height),
        dpi=150,
        facecolor="w",
        edgecolor="k",
    )

    y_pos = np.arange(feature_values.shape[0])
    plt.barh(y_pos, feature_values.iloc[:, 1], align="center", alpha=0.75)
    plt.yticks(y_pos, feature_names)
    plt.xlabel("Coefficient")
    plt.title(f"Feature coefficient: {model_name}")

    # Flip axis so highest feature_values are plotted on top of figure
    axes = plt.gca()
    axes.invert_yaxis()

    # Remove ticks on y-axis
    axes.tick_params(axis="both", which="both", length=0)

    # Remove top, right, and left border
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_visible(False)

    # Auto-adjust layout and whitespace
    plt.tight_layout()

    fpath = os.path.join(plot_path, f"feature_coefficients_{model_name}{image_ext}")

    plt.savefig(fpath, bbox_inches="tight", pad_inches=0.01)
    plt.close()
    logging.info(f"Saved {fpath}")
