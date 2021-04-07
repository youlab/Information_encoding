from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from PIL import Image


def img_resize(in_path: str, out_path: str, image_size: int) -> None:
    """ Reads an image from `in_path`, resizes it, and saves it at `out_path`"""
    in_img = Image.open(in_path)
    out_img = in_img.resize((image_size, image_size), Image.LANCZOS)
    out_img.save(out_path)


def binary_label_to_decimal(labels: np.ndarray) -> np.ndarray:
    """ Turns a list of binary vectors to their decimal format."""
    output = np.zeros(labels.shape[0])
    for i, x in enumerate(labels):
        output[i] = np.argmax(x)
    return output.astype(int)


def label_distribution(data_frame: pd.DataFrame, column_target: str) -> List[Tuple[int, int]]:
    """
    Provides the distribution information on a target column of a data frame
    """
    unique_tags = data_frame[column_target].unique()
    output = []
    for tag in unique_tags:
        output.append((tag, len(data_frame[data_frame[column_target] == tag])))
    output = sorted(output, key=lambda item: item[0])
    return output


def prediction_standardized(predictions: np.ndarray) -> np.ndarray:
    """
        Turns the predictions to a standard binary output so we can do performance calculations.
    """
    output = np.zeros(predictions.shape)
    for i, prediction in enumerate(predictions):
        output[i, np.argmax(prediction)] = 1
    return output


def label_str_to_dec(labels: np.ndarray, conversion_list: Dict[Union[str, int], int]) -> np.ndarray:
    # Convert string labels into numbers
    output = np.zeros(labels.shape[0])
    for i, x in enumerate(labels):
        output[i] = conversion_list[x]
    return output.astype(int)

# Convert binary labels to decimal
def label_binary_to_dec(labels):
    output=np.zeros(labels.shape[0])
    for i,x in enumerate(labels):
        output[i]=np.argmax(x)
    return output.astype(int)

def aggregate_generator_labels(data_generator) -> np.ndarray:
    output = []
    for batch in data_generator:
        output.extend(batch[1])
    output = np.asarray(output, dtype="int8")
    return output
