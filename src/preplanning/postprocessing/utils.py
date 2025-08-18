import numpy as np
from typing import Any, Dict, List, Tuple


def find_absolute_min_max(
    list_of_dicts: List[Dict[str, List[float]]], key: str
) -> Tuple[float, float]:
    """Find the absolute minimum and maximum values of a key across multiple dictionaries.

    Args:
        list_of_dicts (List[Dict[str, List[float]]]): 
            A list of dictionaries, where each dictionary may contain a list of numeric values under `key`.
        key (str): 
            The key whose values are checked for min/max.

    Returns:
        Tuple[float, float]: 
            The overall minimum and maximum values across all dictionaries containing the key.
    """
    min_key_values = min(min(d[key]) for d in list_of_dicts if key in d)
    max_key_values = max(max(d[key]) for d in list_of_dicts if key in d)
    return min_key_values, max_key_values


def find_strategy_arg_min_max(
    list_of_dicts: List[Dict[str, List[float]]], key: str
) -> Tuple[List[int], List[int]]:
    """Find the indices of minimum and maximum values for a key across multiple dictionaries.

    Args:
        list_of_dicts (List[Dict[str, List[float]]]): 
            A list of dictionaries, where each dictionary may contain a list of numeric values under `key`.
        key (str): 
            The key whose values are checked for argmin/argmax.

    Returns:
        Tuple[List[int], List[int]]: 
            Two lists: the first contains indices of the minimum values for each dictionary, 
            the second contains indices of the maximum values.
    """
    min_key_values = [np.argmin(d[key]) for d in list_of_dicts if key in d]
    max_key_values = [np.argmax(d[key]) for d in list_of_dicts if key in d]
    return min_key_values, max_key_values


def filter_dict_by_prefix(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Filter a dictionary to include only keys that start with a given prefix.

    Args:
        d (Dict[str, Any]): 
            Input dictionary.
        prefix (str): 
            Prefix to match (compared against the first characters of each key).

    Returns:
        Dict[str, Any]: 
            A filtered dictionary containing only items where the key starts with the given prefix.
    """
    return {k: v for k, v in d.items() if k[: len(prefix)] == prefix}
