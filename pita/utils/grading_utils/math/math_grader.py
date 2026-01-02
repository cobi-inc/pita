"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
import re
import sympy
from typing import List, Optional
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

import math_normalize as math_normalize


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str) -> sympy.Expr:
    """
    Parse an expression with sympy.

    Converts mathematical expressions to sympy objects, handling implicit
    multiplication and converting caret notation to Python exponentiation.

    Args:
        expr: A mathematical expression string.

    Returns:
        A sympy expression object.

    Raises:
        Exception: If the expression cannot be parsed by sympy.
    """
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """
    Parse LaTeX to an expression sympy can read.

    Converts LaTeX mathematical notation to plain text that can be processed
    by sympy, including handling special characters and mathematical symbols.

    Args:
        expr: A LaTeX mathematical expression string.

    Returns:
        A plain text mathematical expression string.
    """
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    """
    Check if a string can be converted to a float.

    Args:
        num: A string to test for float conversion.

    Returns:
        True if the string can be converted to a float, False otherwise.
    """
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    """
    Check if a float is effectively an integer.

    Determines if a float value is within 1e-7 of an integer value.

    Args:
        x: A float value to test.

    Returns:
        True if the value is within 1e-7 of an integer, False otherwise.
    """
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    """
    Check if a string represents a fraction in the form a/b.

    Args:
        expr: A string to test for fraction format.

    Returns:
        True if the string matches the pattern of a fraction (e.g., "3/4", "-5/2"),
        False otherwise.
    """
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    """
    Check if a string represents an integer value.

    Handles strings with properly formatted commas and checks if the
    resulting float is within 1e-7 of an integer.

    Args:
        x: A string to test for integer representation.

    Returns:
        True if the string represents an integer value, False otherwise.
    """
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> int:
    """
    Convert a string to an integer.

    Removes commas from the string and converts to integer.

    Args:
        x: A string representation of a number.

    Returns:
        The integer value of the string.

    Raises:
        ValueError: If the string cannot be converted to a number.
    """
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str) -> str:
    """
    Convert implicit mixed numbers to evaluable expressions.

    Transforms mixed numbers with implicit addition to explicit addition.
    For example, "7 3/4" becomes "7+3/4".

    Args:
        step: A string that may contain mixed numbers.

    Returns:
        The string with mixed numbers converted to explicit addition.
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str) -> str:
    """
    Remove properly formatted thousand-separator commas from numbers.

    Strips commas that appear in the standard format (e.g., "1,000,000")
    while being careful not to remove commas that are part of tuple notation.

    Args:
        expr: A string that may contain numbers with comma separators.

    Returns:
        The string with thousand-separator commas removed.
    """
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> Optional[str]:
    """
    Normalize answer expressions.

    Performs comprehensive normalization including:
    - Removing text wrappers and special characters
    - Converting units and large number words
    - Parsing LaTeX notation
    - Handling mixed numbers and negative signs
    - Case normalization

    Args:
        expr: A mathematical expression string, potentially in LaTeX format.

    Returns:
        A normalized expression string, or None if the input is None.
    """
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(fr"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(fr"\^ *\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str) -> int:
    """
    Count the number of unknown letters in an expression.

    Removes known mathematical function names (sqrt, frac) and counts
    the remaining alphabetic characters.

    Args:
        expr: A mathematical expression string.

    Returns:
        The number of distinct unknown alphabetic characters in the expression.
    """
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str) -> bool:
    """
    Determine if an expression is safe to evaluate with sympy.

    Checks for potentially problematic patterns that might cause sympy to hang
    or fail, including too many unknown variables and known bad patterns.

    Args:
        expr: A mathematical expression string.

    Returns:
        True if the expression is safe to evaluate, False otherwise.
    """
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    """
    Check if two expressions are mathematically equivalent using sympy.

    Subtracts the two expressions and simplifies the result. If the
    simplified difference equals zero, the expressions are equivalent.

    Args:
        ground_truth_normalized: The normalized ground truth expression.
        given_normalized: The normalized given expression to check.

    Returns:
        True if the expressions are mathematically equivalent, False otherwise.
    """
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str) -> List[str]:
    """
    Split the elements in a tuple or interval.

    Handles well-formatted commas in large numbers while splitting tuple
    elements. Recognizes tuples by their bracketing characters.

    Args:
        expr: A string representing a tuple, interval, or single value.

    Returns:
        A list of string elements. If the expression is a tuple/interval,
        returns the individual elements; otherwise returns a list containing
        the original expression.
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    Grade a given answer against the ground truth.

    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0

    Special handling for:
    - Tuples and intervals (must match structure and order)
    - Fractions (must be in reduced form)
    - Integers (must be exact matches, no decimal equivalents)

    Args:
        given_answer: The answer to grade.
        ground_truth: The correct answer to compare against.

    Returns:
        True if the answer is correct, False otherwise.
    """
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


if __name__ == "__main__":
    # pass
    answer = grade_answer("1.5", "1/2")
    print("answer: ", answer)