import numpy as np


def extract_collector_vision(layer, x, y, subarray_size):
    # Calculate the half-size of the subarray
    # NOTE sub_array_size MUST BE AN ODD NUMBER!
    half_size = subarray_size // 2

    # Calculate the boundaries of the subarray
    start_x = x - half_size
    end_x = x + half_size + 1
    start_y = y - half_size
    end_y = y + half_size + 1

    # Initialize the subarray with -1 values
    subarray = np.full((subarray_size, subarray_size), 0, dtype=int)

    # Calculate the valid indices for the layer array
    layer_start_x = max(0, start_x)
    layer_end_x = min(layer.shape[0], end_x)
    layer_start_y = max(0, start_y)
    layer_end_y = min(layer.shape[1], end_y)

    # Calculate the corresponding indices for the subarray
    subarray_start_x = layer_start_x - start_x
    subarray_end_x = subarray_start_x + (layer_end_x - layer_start_x)
    subarray_start_y = layer_start_y - start_y
    subarray_end_y = subarray_start_y + (layer_end_y - layer_start_y)

    subarray[subarray_start_x:subarray_end_x, subarray_start_y:subarray_end_y] = \
        layer[layer_start_x:layer_end_x, layer_start_y:layer_end_y]

    return subarray


# Basic normalizer - does no safety checks
def normalize_value(value, min_value, max_value) -> float:
    normalized = (value - min_value) / (max_value - min_value)
    return normalized
