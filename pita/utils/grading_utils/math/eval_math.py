import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from math_grader import grade_answer
from parse_utils import parse_answer
import argparse


def safe_grade(ans: str, correct_ans: str) -> int:
    """
    Safely grade an answer against the correct answer.

    Attempts to grade the given answer using the grade_answer function.
    Returns 0 if any exception occurs during grading.

    Args:
        ans: The student's answer to grade.
        correct_ans: The correct answer to compare against.

    Returns:
        1 if the answer is correct, 0 if incorrect or if an exception occurred.
    """
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0


def eval_math(fname: str) -> Tuple[int, int, int, int, int]:
    """
    Evaluate math answers from a CSV file using different sampling methods.

    Reads a CSV file containing answers from different sampling strategies
    and grades them against the correct answers.

    Args:
        fname: Path to the CSV file containing answers and correct solutions.

    Returns:
        A tuple containing (naive_sampling_correct, low_temp_sampling_correct,
        power_sampling_sliding_window_correct, power_sampling_correct, total),
        where each value represents the count of correct answers for that method
        and total is the number of questions evaluated.
    """
    print(fname)
    df = pd.read_csv(fname)
    naive_sampling_correct = 0
    low_temp_sampling_correct = 0
    power_sampling_sliding_window_correct = 0
    power_sampling_correct = 0
    total = len(df)

    for i in range(total):
        naive_sampling_correct += safe_grade(df["naive_sampling_answer"][i], df["correct_answer"][i])
        low_temp_sampling_correct += safe_grade(df["low_temp_sampling_answer"][i], df["correct_answer"][i])
        power_sampling_sliding_window_correct += safe_grade(df["power_sampling_windowed_answer"][i], df["correct_answer"][i])
        power_sampling_correct += safe_grade(df["power_sampling_answer"][i], df["correct_answer"][i])


    return naive_sampling_correct, low_temp_sampling_correct, power_sampling_sliding_window_correct, power_sampling_correct, total


def math_results(fnames: List[str]) -> Dict[str, float]:
    """
    Compute and display aggregate math results across multiple CSV files.

    Evaluates answers from multiple CSV files using different sampling strategies
    and computes accuracy metrics for each strategy.

    Args:
        fnames: List of paths to CSV files containing answers and correct solutions.

    Returns:
        A dictionary containing accuracy metrics for each sampling strategy:
        - naive_sampling_acc: Accuracy of naive sampling method.
        - low_temp_sampling_acc: Accuracy of low temperature sampling method.
        - power_sampling_sliding_window_acc: Accuracy of power sampling with sliding window.
        - power_sampling_acc: Accuracy of power sampling method.
    """
    naive_sampling_total = 0
    low_temp_sampling_total = 0
    power_sampling_sliding_window_total = 0
    power_sampling_total = 0
    total = 0

    for fname in fnames:
        naive, low_temp, power_sliding, power, n = eval_math(fname)
        naive_sampling_total += naive
        low_temp_sampling_total += low_temp
        power_sampling_sliding_window_total += power_sliding
        power_sampling_total += power
        total += n

    denom = max(total, 1)
    naive_sampling_acc = naive_sampling_total / denom
    low_temp_sampling_acc = low_temp_sampling_total / denom
    power_sampling_sliding_window_acc = power_sampling_sliding_window_total / denom
    power_sampling_acc = power_sampling_total / denom

    print(f"Files evaluated: {len(fnames)}")
    print(f"Total questions: {total}")
    print(f"Naive sampling accuracy:  {naive_sampling_acc:.3f}")
    print(f"Low temperature sampling accuracy:  {low_temp_sampling_acc:.3f}")
    print(f"Power sampling sliding window accuracy:  {power_sampling_sliding_window_acc:.3f}")
    print(f"Power Sampling accuracy:  {power_sampling_acc:.3f}")

    return {
        "naive_sampling_acc": naive_sampling_acc,
        "low_temp_sampling_acc": low_temp_sampling_acc,
        "power_sampling_sliding_window_acc": power_sampling_sliding_window_acc,
        "power_sampling_acc": power_sampling_acc,
    }

if __name__ == "__main__":
    # read in a csv file
    df = pd.read_csv('AIME_power_sampling_results_temp_0.25_11_12_qwen2_5_MATH_7b.csv')

    # parse columns of the CSV file
    for col in df.columns:
        if col.endswith('_output'):
            new_col = col.replace('_output', '_answer')
            df[new_col] = df[col].apply(parse_answer)

    # Add correctness columns for each answer
    for col in df.columns:
        if col.endswith('_answer') and col != 'correct_answer':
            correct_col = col.replace('_answer', '_correct')
            df[correct_col] = df.apply(lambda row: safe_grade(row[col], row['correct_answer']), axis=1)

    # Create a new CSV file with only the new columns
    new_csv_name = 'MATH500_power_sampling_results_parsed.csv'
    new_cols = [col for col in df.columns if col.endswith('_answer')]
    correct_cols = [col for col in df.columns if col.endswith('_correct')]
    old_cols = [col for col in df.columns if col.endswith('_output')]
    df[new_cols + correct_cols + old_cols].to_csv(new_csv_name, index=False)

    fnames = [new_csv_name]
    math_results(fnames)