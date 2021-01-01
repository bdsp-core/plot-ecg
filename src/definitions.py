PDF_EXT = ".pdf"
TENSOR_EXT = ".hd5"
XML_EXT = ".xml"
MUSE_ECG_XML_MRN_COLUMN = "PatientID"
ECG_PREFIX = "ecg"
ECG_DATE_FORMAT = "%m-%d-%Y"
ECG_TIME_FORMAT = "%H:%M:%S"
ECG_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
ECG_ZERO_PADDING_THRESHOLD = 0.25
ECG_REST_LEADS_ALL = {
    "I": 0,
    "II": 1,
    "III": 2,
    "aVR": 3,
    "aVL": 4,
    "aVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
}
ECG_REST_LEADS_INDEPENDENT = {
    "I": 0,
    "II": 1,
    "V1": 2,
    "V2": 3,
    "V3": 4,
    "V4": 5,
    "V5": 6,
    "V6": 7,
}
