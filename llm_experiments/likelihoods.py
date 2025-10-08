import os


from contextlib import nullcontext
from glob import glob
import json
import random
from tqdm import tqdm
import argparse

import pandas as pd
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from datasets import Dataset, load_dataset, concatenate_datasets


import torch
import torch.nn as nn
from torch.nn import functional as F
import transformers

from utils.parse_utils import parse_answer
from constants import *
import pandas as pd
from grader_utils.math_grader import grade_answer


def log_probs(p, sequence, prefix_len):
    # sequence is tokenized w.r.t. base_tokenizer, so split into context and gen
    ids = torch.tensor(sequence, dtype=torch.long, device=p.device).unsqueeze(0)  # [1, C+G]
    
    with torch.no_grad():
        out = p(input_ids=ids)
    logits = out.logits[0, prefix_len-1:-1, :]        

    probs_seq = F.log_softmax(logits, dim=-1)
    exp_probs = probs_seq.exp()

    gen_ids = ids[0, prefix_len:]
    log_probs = probs_seq.gather(dim=1, index=gen_ids.unsqueeze(1)).view(-1)

    log_likelihood = log_probs.mean().item()
    confidence = 1/len(gen_ids)*(exp_probs * probs_seq).sum(()).item()
    return log_likelihood, confidence


model = "Qwen/Qwen2.5-Math-7B"
device = torch.device("cuda:0")
tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code = True)
p = transformers.AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, trust_remote_code = True).to(device)
p.eval()


mcmc_seq_list = []
std_seq_list = []
mcmc_question = []

for j in [10]:
    naive_total = 0
    for idx in range(5):
        i_tot = 0
        # construct the filename
        filename = "results/qwen_math/" + "qwen_math_rlcomp_base_low_temp_results_" + str(j) + "_0.25_" + str(idx) + "_7" + ".csv"
        # load the dataframe
        df = pd.read_csv(filename)

        for i in range(100):
            try:
                std_seq_list.append(df["question"][i] + df["std_completion"][i])
                mcmc_seq_list.append(df["mcmc_completion"][i])
                mcmc_question.append(df["question"][i])
            except:
                print(df["question"][i])
                print(df["std_completion"][i])

print(len(mcmc_seq_list))

grpo_seq_list = []
grpo_question = []


for j in [10]:
    naive_total = 0
    for idx in range(1):
        filename = "results/qwen_grpo/" + "qwen_grpo_rlcomp_base_low_temp_results_" + str(j) + "_0.25_" + str(idx) + "_0" + ".csv"
        # load the dataframe
        df = pd.read_csv(filename)
        for i in range(500):
            try:
                grpo_seq_list.append(df["question"][i] + df["naive_completion"][i])
                grpo_question.append(df["question"][i])
            except:
                pass

print(len(grpo_seq_list))

mcmc_lps = []
std_lps = []

for idx in tqdm(range(len(mcmc_seq_list))):
    mcmc_seq = mcmc_seq_list[idx]
    std_seq = std_seq_list[idx]
    question = mcmc_question[idx]

    mcmc_tokens = tokenizer.encode(mcmc_seq, add_special_tokens=False)
    std_tokens = tokenizer.encode(std_seq, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)

    prefix_len = len(question_tokens)

    mcmc_lp = log_probs(p, mcmc_tokens, prefix_len)
    std_lp = log_probs(p, std_tokens, prefix_len)
    mcmc_lps.append(mcmc_lp)
    std_lps.append(std_lp)



grpo_lps = []

for idx in tqdm(range(len(grpo_seq_list))):
    grpo_seq = grpo_seq_list[idx]
    question = grpo_question[idx]

    grpo_tokens = tokenizer.encode(grpo_seq, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)

    prefix_len = len(question_tokens)

    grpo_lp = log_probs(p, grpo_tokens, prefix_len)
    grpo_lps.append(grpo_lp)


import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Nimbus Roman",   # change to "Times New Roman" if installed
    "mathtext.fontset": "cm",
})

def first(xs):
    return [float(a[0]) for a in xs
            if a is not None and len(a) >= 1 and not (isinstance(a[0], float) and math.isnan(a[0]))]

def second(xs):
    return [float(a[1]) for a in xs
            if a is not None and len(a) >= 2 and not (isinstance(a[1], float) and math.isnan(a[1]))]

mcmc_ll   = first(mcmc_lps)
std_ll    = first(std_lps)
grpo_ll   = first(grpo_lps)

mcmc_conf = second(mcmc_lps)
std_conf  = second(std_lps)
grpo_conf = second(grpo_lps)

def summarize(name, arr):
    if len(arr) == 0:
        print(f"{name}: (no data)")
        return
    print(f"{name}: n={len(arr)}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}, "
          f"min={np.min(arr):.4f}, max={np.max(arr):.4f}")

print("Log-likelihood stats (avg per token):")
summarize("MCMC", mcmc_ll)
summarize("STD",  std_ll)
summarize("GRPO", grpo_ll)

print("\nConfidence stats (avg per token):")
summarize("MCMC", mcmc_conf)
summarize("STD",  std_conf)
summarize("GRPO", grpo_conf)

# --- Color palette: Blues gradient ---
# Make MCMC darkest; STD/GRPO lighter
palette = {
    "Ours": plt.cm.Blues(0.95),
    "Base":  plt.cm.Blues(0.40),
    "GRPO": plt.cm.Blues(0.75),
}

# Base kwargs; we’ll specialize per-series below
bins = 30
base_kwargs = dict(bins=bins, density=True)

# ===== Histogram 1: Log-likelihoods =====
plt.figure(figsize=(6, 6))

if len(std_ll):
    plt.hist(std_ll, label="Base", color=palette["Base"],
             alpha=0.75, linewidth=0.6, **base_kwargs)
if len(grpo_ll):
    plt.hist(grpo_ll, label="GRPO", color=palette["GRPO"],
             alpha=0.75, linewidth=0.6, **base_kwargs)

if len(mcmc_ll):
    # Filled bars
    plt.hist(
        mcmc_ll, label="Ours", color=palette["Ours"],
        alpha=0.75, linewidth=0.6, **base_kwargs
    )

plt.xlabel("Average Log-Likelihood", fontsize=19)
plt.ylabel("Density", fontsize=19)

handles, labels = plt.gca().get_legend_handles_labels()
order = [labels.index("Ours")] + [i for i, l in enumerate(labels) if l != "Ours"]
plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=16)
ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
plt.tight_layout()
plt.savefig("hist_loglikelihoods.png", dpi=600, bbox_inches="tight")
plt.show()

# ===== Histogram 2: Confidence =====
plt.figure(figsize=(6, 6))

if len(std_conf):
    plt.hist(std_conf, label="Base", color=palette["Base"],
             alpha=0.75, linewidth=0.6, **base_kwargs)
if len(grpo_conf):
    plt.hist(grpo_conf, label="GRPO", color=palette["GRPO"],
             alpha=0.75, linewidth=0.6, **base_kwargs)

if len(mcmc_conf):
    plt.hist(
        mcmc_conf, label="Ours", color=palette["Ours"],
        alpha=0.75, linewidth=0.6, **base_kwargs
    )

plt.xlabel("Average Confidence", fontsize=19)
plt.ylabel("Density", fontsize=19)
handles, labels = plt.gca().get_legend_handles_labels()
order = [labels.index("Ours")] + [i for i, l in enumerate(labels) if l != "Ours"]
plt.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=16)
ax = plt.gca()
ax.set_axisbelow(True)
ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
plt.tight_layout()
plt.savefig("hist_confidence.png", dpi=600, bbox_inches="tight")
plt.show()
