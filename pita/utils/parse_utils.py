from typing import Optional, Tuple

def remove_boxed(s: str) -> Optional[str]:
    """
    Remove the LaTeX \\boxed{} wrapper from a string.

    Args:
        s: A string that should start with "\\boxed{" and end with "}".

    Returns:
        The content inside the \\boxed{} wrapper, or None if the string doesn't
        have the expected format.
    """
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except (AssertionError, IndexError, TypeError):
        return None


def last_boxed_only(sample: Tuple[str, str]) -> Optional[Tuple[str, str]]:
    """
    Filter a (question, answer) sample to keep only the last \\boxed{} or \\fbox{} element.

    Given a (q,a) sample, filter the answers so that they only contain
    the last \\boxed{...} or \\fbox{...} element

    Args:
        sample: A tuple of (question, answer) strings.

    Returns:
        A tuple of (question, filtered_answer) where filtered_answer contains only
        the last boxed element, or None if no boxed element is found.
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string: str) -> Optional[str]:
    """
    Extract the last \\boxed{} or \\fbox{} expression with proper brace matching.

    This function handles nested braces correctly by tracking open and closed braces
    to find the complete expression.

    Args:
        string: A string potentially containing \\boxed{} or \\fbox{} expressions.

    Returns:
        The last complete \\boxed{} or \\fbox{} expression (including the command
        and braces), or None if no such expression is found or braces don't match.
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


def parse_answer(input_str: Optional[str]) -> Optional[str]:
    """
    Parse and extract the final answer from a mathematical solution.

    This function finds the last \\boxed{} expression in the answer string and
    extracts the content from within the braces.

    Args:
        input_str: A string containing a mathematical solution, potentially with
            one or more \\boxed{} expressions, or None.

    Returns:
        The extracted answer content (without the \\boxed{} wrapper), or None
        if no boxed answer is found or the input is None.
    """
    if input_str is None:
        return None
    last_boxed = last_boxed_only_string(input_str)
    if last_boxed is None:
        return None
    return remove_boxed(last_boxed)
