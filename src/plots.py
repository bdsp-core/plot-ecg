# Imports: standard library
import os
import re
import math
import argparse
from typing import Dict, List, Tuple, Union, Callable, Optional
from datetime import datetime
from collections import OrderedDict

# Imports: third party
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

# Imports: first party
from definitions import HD5_EXT, ECG_DATE_FORMAT, ECG_DATETIME_FORMAT, ECG_PREFIX

# fmt: off
# need matplotlib -> Agg -> pyplot
import matplotlib      # isort:skip
matplotlib.use("Agg")  # isort:skip
from matplotlib import pyplot as plt  # isort:skip
# fmt: on

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


def _plot_ecg_text(
    data: dict,
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
        f"{data['patientlastname']}, {data['patientfirstname']}",
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


def _hf_to_dict(hf) -> dict:
    keys = [
        "patientid",
        "patientfirstname",
        "patientlastname",
        "gender",
        "dateofbirth",
        "patientage",
        "sitename",
        "location",
        "read_md_clean",
        "taxis_md",
        "ventricularrate_md",
        "printerval_md",
        "qrsduration_md",
        "qtinterval_md",
        "paxis_md",
        "raxis_md",
        "qtcorrected_md",
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    ecg_data_dict = dict()
    for key in keys:
        if key in hf:
            if len(hf[key].shape) == 0:
                ecg_data_dict[key] = hf[key][()]
            else:
                ecg_data_dict[key] = hf[key][:]
        else:
            ecg_data_dict[key] = None

    dt = hf.name.replace("/ecg/", "")
    ecg_data_dict["datetime"] = dt
    return ecg_data_dict


def _plot_ecg_figure(
    ecg_hd5_object: h5py._hl.group.Group,
    output_folder: str,
) -> int:
    ecg_data_dict = _hf_to_dict(hf=ecg_hd5_object)

    breakpoint()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9.5
    w, h = 11, 8.5
    fig = plt.figure(figsize=(w, h), dpi=100)

    _plot_ecg_text(ecg_data_dict, fig, w, h)

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

    title = re.sub(r"[:/. ]", "", f'{patient_id}_{data["datetime"]}')
    fpath = os.path.join(output_folder, f"{title}.png")
    plt.savefig(fpath)
    plt.close(fig)
    return fpath


def plot_ecg(args):
    # Get full paths to all HD5 files
    paths = []
    for root, dirs, files in os.walk(args.hd5):
        for file in files:
            if file.endswith(HD5_EXT):
                path = os.path.join(root, file)
                paths.append(path)

    # Iterate over hd5 files
    num_ecgs = 0
    for path in tqdm(paths):
        with h5py.File(path, "r") as hf:
            ecg_datetimes = list(hf[ECG_PREFIX].keys())
            for ecg_datetime in ecg_datetimes:
                ecg_hd5_object = hf[f"{ECG_PREFIX}/{ecg_datetime}"]
                outcome = _plot_ecg_figure(
                    ecg_hd5_object=ecg_hd5_object, output_folder=args.plot
                )
                num_ecgs += outcome
    print(f"Saved {num_ecgs} ECGs from {len(paths)} HD5 files to {args.plot}")
