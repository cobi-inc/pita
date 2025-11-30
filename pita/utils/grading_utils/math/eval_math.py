import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any
from math_grader import grade_answer
from parse_utils import parse_answer
import argparse


def safe_grade(ans, correct_ans):
    try:
        return int(grade_answer(ans, correct_ans))
    except Exception:
        return 0


def eval_math(fname):
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


def math_results(fnames):
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