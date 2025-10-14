import os

from contextlib import nullcontext
from glob import glob
import json
import random
import time
from tqdm import tqdm
import argparse
import datasets
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

from constants import *

def parse_answer(text):
    """
    Parse the final answer from generated text.
    Looks for answers in \\boxed{} format and extracts the content.
    
    Args:
        text (str): The generated text containing the answer
        
    Returns:
        str: The extracted answer, or "No answer found" if no boxed answer is found
    """
    import re
    
    # Look for \boxed{...} pattern
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(boxed_pattern, text)
    
    if matches:
        # Return the last boxed answer found (in case there are multiple)
        return matches[-1].strip()
    
    # If no boxed answer found, look for other common answer patterns
    # Look for "The answer is ..." or "Answer: ..." patterns
    answer_patterns = [
        r'(?:The answer is|Answer:)\s*([^\n.]+)',
        r'(?:Final answer|Final result):\s*([^\n.]+)',
        r'(?:Therefore|So),?\s*([^\n.]+)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    
    # If still no answer found, try to extract the last line that looks like an answer
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith(('Step', 'First', 'Next', 'Then', 'Finally', 'Therefore', 'So')):
            # Check if it looks like a mathematical answer
            if any(char.isdigit() or char in '+-*/=()[]{}' for char in line):
                return line
    
    return "No answer found"

class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    # returns log probs
    @torch.no_grad()
    def next_token(self, prefix):
        device = self.device
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size:]
        output = self.model(prefix_cond)
        logits = output.logits
        logits = logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)

def naive_temp(p : AutoregressiveSampler, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]


    idx = tokens.view(unscaled_logits.shape[0], 1, 1)

    log_probs_unnorm = (1/temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)

    return prop, log_probs_norm, log_probs_unnorm

def mcmc_temp(p : AutoregressiveSampler, context, temp, mcmc_steps, max_new_tokens, block_num=16, sliding_window=0):
    c = len(context)
    print(f'Temp: {temp}')
    gen = []
    if context is not None:
        gen = context.copy()
    log_probs_norm = []
    log_probs_unnorm = []


    print(max_new_tokens)
    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
    print(jump_size)
    attempts = 0
    acceptances = 0


    for block_count in tqdm(range(block_num)):
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size+len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        for _ in tqdm(range(mcmc_steps)):
            attempts+=1
            t = len(gen)
            context_start = c + block_count * jump_size * sliding_window
            if(context_start >= t):
                context_start = t-jump_size

            print(f"Sampling index in range ({context_start}, {t-1})")
            idx = random.randint(context_start,t-1)
            # llm query takes the burden of time 
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert(len(log_prob_prop) == s - idx)
            assert(len(target_log_prob_prop) == s - idx)
            log_prob_cur = log_probs_norm.copy()[idx-c:s-c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx-c:s-c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances+=1
                gen = prop.copy()
                log_probs_norm[idx-c:] = log_prob_prop.copy()
                log_probs_unnorm[idx-c:] = target_log_prob_prop.copy()

                del prop
                del log_prob_prop
                del target_log_prob_cur

        if p.tokenizer.eos_token_id in gen:
            eos_idx = gen.index(p.tokenizer.eos_token_id)
            gen = gen[:eos_idx + 1]
            log_probs_norm = log_probs_norm[:eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[:eos_idx + 1]
            acceptance_ratio = acceptances/attempts
            return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances/attempts
    return gen, log_probs_norm, log_probs_unnorm, acceptance_ratio

def format_prompt(question, model, tokenizer, cot=True):
    if model == "qwen":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_math":
        format_str = PROMPT + question
        if cot:
            format_str+=COT
        else:
            format_str+=BASE

    elif model == "qwen_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "qwen_math_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi_grpo":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "phi":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    elif model == "tulu":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
    
    elif model == "gptoss_high":
        content_str = PROMPT + question
        if cot:
            content_str+=COT
        else:
            content_str+=BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)

    return format_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", action = "store", type = str, default = "results/",  dest = "save_str")
    parser.add_argument("--model", action = "store", default = "qwen", type = str, choices = ["qwen", "qwen_math", "phi", "tulu", "qwen_grpo", "qwen_math_grpo", "phi_grpo","gptoss_high"])
    parser.add_argument("--temperature", action = "store", default = 0.5, type = float, dest = "temperature")
    parser.add_argument("--dataset", action = "store", default = "MATH", type = str)
    parser.add_argument("--cot", action = "store", type = bool, default = True)
    parser.add_argument("--mcmc_steps", action = "store", type = int, default = 10)
    parser.add_argument("--device", action = "store", type = str, dest = "device", default = "cuda" if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--batch_idx", action = "store", type = int, default = 0)
    parser.add_argument("--seed", action = "store", type = int, default = 0)
    args = parser.parse_args()

    random.seed(0)


    model = args.model
    device = args.device
    dataset_name = args.dataset
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps

    save_str = os.path.join(args.save_str, model)
    os.makedirs(save_str, exist_ok=True)


    print(model)
    print(device)
    print(mcmc_steps)
    if model == "qwen":
        model_str = "Qwen/Qwen2.5-7B"
    elif model == "qwen_math":
        model_str = "Qwen/Qwen2.5-Math-7B"
    elif model == "qwen_grpo":
        model_str = "/net/holy-isilon/ifs/rc_labs/ydu_lab/aakaran/models/grpo"
    elif model == "qwen_math_grpo":
        model_str = "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150"
    elif model == "phi":
        model_str = 'microsoft/Phi-3.5-mini-instruct'
    elif model == "tulu":
        model_str = "allenai/Llama-3.1-Tulu-3-8B-DPO"
    elif model == "gptoss_high":
        model_str= "openai/gpt-oss-20b"

    if dataset_name == "MATH":
        # dataset_str = "EleutherAI/hendrycks_math"
        ##json_file = 'MATH-TTT.json'
        ##dataset = json.load(open(json_file, "r"))
        # random.seed(0)
        # random.shuffle(dataset)
        ds = datasets.load_dataset("MathArena/AIME_2025")["train"]
        old_column_names = ds.column_names
        ds = ds.map(lambda x: {k.lower(): v for k, v in x.items()})
        # use existing id as unique_id
        ds = ds.rename_column('problem_idx', 'unique_id')
        # convert answer to string type
        dataset = ds.cast_column('answer', datasets.Value('string'))



    print("dataset done")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code = True)
    # hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype=torch.float32, trust_remote_code = True).to(device)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str, torch_dtype="auto", device_map="auto", trust_remote_code = True)
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, device)

    print("loaded models")
    results = []

    start = 100*args.batch_idx
    end = 100*(args.batch_idx+1)
    # start = 0
    # end = len(dataset)
    # start = 0
    # end = 1


    for idx, x in tqdm(enumerate(dataset), desc = "Benchmark on MATH"):
        # Only try the first question of the MATH Dataset Benchmar
        if(idx == 1):
            break

        print(x)
        question = x["problem"]
        print(question)
        answer = x["answer"]
        print(answer)

        input_text = format_prompt(question, model, tokenizer, cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        prefx = [idx.item() for idx in input_ids[0]]

        # naive_temp_output = hf_model.generate(input_ids, max_new_tokens=3072, 
        #                         return_dict_in_generate=True, output_scores=True, temperature = temp)
        
        # print(tokenizer.decode(naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        # print("naive done")
        
        
        # std_output = hf_model.generate(input_ids, max_new_tokens=3072, 
        #                         return_dict_in_generate=True, output_scores=True, do_sample = True)
        
        # print(tokenizer.decode(std_output[0][:, len(input_ids[0]):].squeeze().to("cpu"), skip_special_tokens=True))
        # print("std done")

        # mcmc_temp_output, _, _, acceptance_ratio = mcmc_temp(autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072, sliding_window=0)
        mcmc_temp_output_sliding, _, _, acceptance_ratio_sliding = mcmc_temp(autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=3072, sliding_window=1)

        # print(len(std_output))
        # print(len(naive_temp_output))
        # print(len(mcmc_temp_output))
        # print(tokenizer.decode(torch.tensor([mcmc_temp_output], dtype=torch.long, device=device).squeeze().to("cpu"), skip_special_tokens=True))
        print(len(mcmc_temp_output_sliding))
        print(tokenizer.decode(torch.tensor([mcmc_temp_output_sliding], dtype=torch.long, device=device).squeeze().to("cpu"), skip_special_tokens=True))
        print("mcmc done")

        # naive_generated_ids = naive_temp_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        # std_generated_ids = std_output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        # mcmc_temp_ids = torch.tensor([mcmc_temp_output], dtype=torch.long, device=device).squeeze().to("cpu")
        mcmc_temp_ids_sliding = torch.tensor([mcmc_temp_output_sliding], dtype=torch.long, device=device).squeeze().to("cpu")

        # naive_completion = tokenizer.decode(naive_generated_ids, skip_special_tokens=True)
        # std_completion = tokenizer.decode(std_generated_ids, skip_special_tokens=True)
        # mcmc_completion = tokenizer.decode(mcmc_temp_ids, skip_special_tokens=True)
        mcmc_completion_sliding = tokenizer.decode(mcmc_temp_ids_sliding, skip_special_tokens=True)

        # naive_answer = parse_answer(naive_completion)
        # std_answer = parse_answer(std_completion)
        # mcmc_answer = parse_answer(mcmc_completion)
        mcmc_answer_sliding = parse_answer(mcmc_completion_sliding)

        # print(naive_answer)
        # print(std_answer)
        # print(mcmc_answer)
        print(mcmc_answer_sliding)
        print(acceptance_ratio_sliding)
        # print(f'Acceptance: {acceptance_ratio}')


        results.append({
            "question": question,
            "correct_answer": answer,
            # "naive_completion": naive_completion,
            # "naive_answer": naive_answer,
            # "std_completion": std_completion,
            # "std_answer": std_answer,
            # "mcmc_completion": mcmc_completion,
            # "mcmc_answer": mcmc_answer,
            "mcmc_completion_sliding": mcmc_completion_sliding,
            "mcmc_answer_sliding": mcmc_answer_sliding
        })

    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_str, model+"_base_power_samp_results_" + str(mcmc_steps) + "_" + str(temp) + "_" + str(args.batch_idx)  + "_" + str(args.seed) + " " + str(time.time()) + ".csv"), index=False)