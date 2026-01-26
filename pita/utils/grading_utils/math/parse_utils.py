import json
import numpy as np
from typing import Optional, Tuple

def remove_boxed(s: str) -> Optional[str]:
    """
    Remove the LaTeX boxed wrapper from a string.

    Extracts the content from within a \\boxed{...} LaTeX command.

    Args:
        s: A string containing a LaTeX \\boxed{} command.

    Returns:
        The content within the boxed command, or None if the string
        doesn't match the expected format.
    """
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def last_boxed_only(sample: Tuple[str, str]) -> Optional[Tuple[str, str]]:
    """
    Extract only the last boxed answer from a question-answer sample.

    Given a (q,a) sample, filter the answers so that they only contain
    the last \\boxed{...} or \\fbox{...} element.

    Args:
        sample: A tuple containing (question, answer) strings.

    Returns:
        A tuple containing (question, last_boxed_answer), or None if no
        boxed element is found in the answer.
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last boxed answer from a string.

    Finds the last occurrence of \\boxed{...} or \\fbox{...} in the string
    and extracts the complete boxed expression, properly handling nested braces.

    Args:
        string: A string potentially containing LaTeX boxed commands.

    Returns:
        The last complete boxed expression (including the \\boxed{} wrapper),
        or None if no boxed element is found or if braces are unbalanced.
    """
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def parse_answer(input_str: str) -> Optional[str]:
    """
    Parse the final answer from a LaTeX string.

    Extracts the content of the last \\boxed{...} or \\fbox{...} command
    in the input string, removing the wrapper.

    Args:
        input_str: A string containing LaTeX formatted mathematical content,
            potentially with multiple boxed expressions.

    Returns:
        The content of the last boxed expression without the wrapper,
        or None if no boxed element is found.
    """
    return remove_boxed(last_boxed_only_string(input_str))
