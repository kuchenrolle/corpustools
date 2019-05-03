import numpy as np


def median_element(array):
    """Identifies the element that comes closest to having
    half of the probability mass to its left and right.

    Parameters
    ----------
    array: np.array
        Probabilities, cumulative from left to right
    """
    mid = (array[0] + array[-1]) / 2
    mid_idx = np.searchsorted(array, mid)

    if (array[mid_idx] - mid) >= (mid - array[mid_idx - 1]):
        return mid_idx - 1

    return mid_idx


def recursive_median(array):
    """Returns indices for recursive median splits of an array.

    Parameters
    ----------
    array : np.array
        Probabilities, cumulative from left to right

    Notes
    -----
    Returns the element that comes closest to the median
    the array, then the elements that comes closest to the median of each
    half of the array split on the first element and so forth.
    """
    if len(array) == 3:
        yield 1
        yield 0
        yield 2

    elif len(array) == 2:
        yield 0
        yield 1

    elif len(array) == 1:
        yield 0

    else:
        median_idx = median_element(array)
        yield median_idx

        for idx in recursive_median(array[:median_idx]):
            yield idx
        for idx in recursive_median(array[median_idx + 1:]):
            yield median_idx + idx + 1


def median_split_vocabulary(frequencies):
    """Returns optimal insertion order for balanced Ternary Search Tree.

    Parameters
    ----------
    frequencies : Counter
        strings to be inserted into a tree with their frequencies

    Notes
    -----
    The optimal insertion order of strings into a ternary search tree
    is such that, the median element is inserted first, then the median
    elements of the two remaining sub-arrays (left and right of the inserted
    median) inserted and so on.
    This function returns strings from a counter in this order, taking
    each strings frequency into account.
    """
    # sort alphabetically
    strings = sorted(frequencies.keys())

    # turn frequencies into cumulative probabilities
    frequencies = np.array([frequencies[string] for string in strings])
    frequencies = frequencies / frequencies.sum()
    frequencies = np.cumsum(frequencies)

    # yield median split indices
    indices = recursive_median(frequencies)
    for idx in indices:
        yield strings[idx]
