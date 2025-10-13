# Software Hierarchy - Training-Free Reasoning

## Overview
This repository implements training-free reasoning methods for language models, with two main components:
1. **Toy Composition**: A simplified autoregressive distribution composition framework
2. **LLM Experiments**: Full-scale language model sampling and evaluation

---

## Core Modules

### 1. Toy Composition (`toy_composition.py`)
Demonstrates compositional sampling with simple autoregressive distributions.

#### Distribution Functions
- **`error_token_p(prefix, T)`**
  - Defines probability distribution p over tokens {1, 2, ..., T, E}
  - If 'E' in prefix → next token is always '1' (prob=1.0)
  - Otherwise → uniform distribution with p(E) = 1/(T+1)

- **`error_token_q(prefix, T)`**
  - Defines probability distribution q with exponentially low prob for '1'
  - p('1') = exp(-T) (alpha)
  - All other tokens get uniform prob = (1-alpha)/T

#### Utility Functions
- **`normalize(dist)`**
  - Normalizes probability distribution to sum to 1
  - Input: dictionary of {token: unnormalized_prob}
  - Output: dictionary of {token: normalized_prob}

- **`set_product(output_p, output_q)`**
  - Computes element-wise product of two distributions
  - Used for composing distributions: p(x) * q(x)

- **`sample_autoregressive(next_token_dist, T, seq_len=None)`**
  - Generates sequence by sampling from autoregressive distribution
  - Parameters:
    - `next_token_dist`: function returning next token distribution
    - `T`: vocabulary size parameter
    - `seq_len`: length of sequence to generate (default: 2*T)

#### Sampling Algorithms

- **`naive_composition(p, q, T, context=None, seq_len=None)`**
  - Naive compositional sampling using normalized product of p and q
  - Returns: (sequence, log_probs_normalized, log_probs_unnormalized)
  - Algorithm:
    1. For each position, compute p(·|prefix) * q(·|prefix)
    2. Normalize and sample
    3. Track both normalized and unnormalized log probabilities

- **`compositional_sampler(p, q, mcmc_steps, T, context=[], seq_len=None)`**
  - MCMC-enhanced compositional sampling
  - Main algorithm:
    1. Generate token using naive composition
    2. Perform `mcmc_steps` Metropolis-Hastings updates:
       - Choose random position idx in [context_len, current_len)
       - Propose resample from idx to end
       - Accept/reject based on Metropolis-Hastings ratio
  - Acceptance criterion: `log_r = target_log_prob_proposed + proposal_log_prob_current - target_log_prob_current - proposal_log_prob_proposed`

---

### 2. LLM Experiments Module (`llm_experiments/`)

#### 2.1 Core Sampling (`power_samp.py`)
Main module for temperature-based sampling with MCMC refinement.

##### Classes
- **`AutoregressiveSampler`**
  - **`__init__(self, model, tokenizer, device)`**: Initialize with HuggingFace model
  - **`next_token(self, prefix)`**: Get log probabilities for next token given prefix
    - Handles context length limits (block_size)
    - Returns log probabilities for all tokens in vocabulary

##### Distribution Operations
- **`normalize(dist)`**
  - Converts logits to normalized probabilities using softmax
  
- **`dist_product(logit_p, logit_q)`**
  - Composes two distributions in log space: logit_p + logit_q

- **`dist_temp_scale(logit_p, temp)`**
  - Applies temperature scaling: logit_p / temp
  - Higher temp → more uniform, lower temp → sharper

##### Sampling Functions

- **`naive_temp(p, context, temp, seq_len)`**
  - Standard temperature-scaled generation
  - Uses HuggingFace `generate()` with temperature parameter
  - Returns: (tokens, log_probs_normalized, log_probs_unnormalized)
  - Tracks both:
    - Normalized probs (after temperature scaling)
    - Unnormalized probs (before temperature scaling)

- **`max_swap(p, context, temp, mcmc_steps, max_new_tokens, block_num=16)`**
  - MCMC sampling with greedy acceptance (accept if log_r > 0)
  - Algorithm:
    1. Generate in blocks of size `max_new_tokens/block_num`
    2. After each block, perform MCMC refinement for `mcmc_steps`:
       - Randomly select position idx
       - Propose regeneration from idx to end
       - Accept if target_log_prob_proposed > target_log_prob_current (greedy)
    3. Early stopping if EOS token generated
  - Returns: (tokens, log_probs_norm, log_probs_unnorm, acceptance_ratio)

- **`mcmc_temp(p, context, temp, mcmc_steps, max_new_tokens, block_num=16)`**
  - MCMC sampling with probabilistic acceptance (Metropolis-Hastings)
  - Same structure as `max_swap` but uses full MH acceptance:
    - `log_r = target_proposed + proposal_current - target_current - proposal_proposed`
    - Accept with probability min(1, exp(log_r))
  - Returns: (tokens, log_probs_norm, log_probs_unnorm, acceptance_ratio)

##### Prompt Formatting
- **`format_prompt(question, model, tokenizer, cot=True)`**
  - Formats prompts for different models:
    - "qwen", "qwen_math": Direct string formatting
    - "qwen_grpo", "qwen_math_grpo", "phi_grpo", "phi", "tulu", "gptoss_high": Chat template formatting
  - Appends chain-of-thought (COT) or base instruction
  - Returns tokenizable prompt string

##### Answer Parsing
- **`parse_answer(text)`**
  - Extracts final answer from generated text
  - Priority order:
    1. `\boxed{...}` pattern (LaTeX)
    2. "The answer is..." / "Answer:" patterns
    3. "Final answer:" / "Therefore" patterns
    4. Last line containing digits/math symbols
  - Returns: extracted answer string or "No answer found"

#### 2.2 Likelihood Analysis (`likelihoods.py`)
Analyzes log-likelihoods and confidence of generated sequences.

##### Functions
- **`log_probs(p, sequence, prefix_len)`**
  - Computes log-likelihood and confidence for a sequence
  - Algorithm:
    1. Forward pass through model with full sequence
    2. Extract logits for positions [prefix_len-1:-1]
    3. Compute log probabilities and gather for actual tokens
    4. Calculate metrics:
       - `log_likelihood = mean(log_probs)`
       - `confidence = (1/len) * sum(exp_probs * log_probs)`
  - Returns: (log_likelihood, confidence)

##### Visualization
- Generates histograms comparing:
  - MCMC samples vs Standard samples vs GRPO samples
  - Metrics: log-likelihood and confidence distributions
- Uses matplotlib with custom styling (Nimbus Roman font)

#### 2.3 Grading Utilities (`grader_utils/`)

##### Math Answer Grading (`math_grader.py`)
- **`grade_answer(given_answer, ground_truth)`**
  - Main grading function
  - Multi-stage comparison:
    1. MATH dataset normalization (via `math_normalize`)
    2. Custom normalization
    3. Tuple/interval element-wise comparison
    4. Symbolic equality check via sympy
  - Returns: True if correct, False otherwise

- **`normalize(expr)`**
  - Comprehensive answer normalization:
    - Removes LaTeX formatting (\text{}, \$, \%, etc.)
    - Converts units (million → *10^6, etc.)
    - Strips unit names (cm, meter, degree, etc.)
    - Handles fractions and mixed numbers
    - Case-insensitive comparison

- **`are_equal_under_sympy(ground_truth_normalized, given_normalized)`**
  - Uses sympy for symbolic math comparison
  - Computes difference and checks if simplifies to 0
  - Safety checks to avoid hanging on complex expressions

- **`split_tuple(expr)`**
  - Parses tuple/interval answers
  - Handles comma-separated elements in parentheses/brackets

##### Math Normalization (`math_normalize.py`)
- **`normalize_answer(answer)`**
  - Hendrycks MATH dataset normalization
  - Removes `\text{}` enclosures
  - Calls `_strip_string()` for further processing

- **`_strip_string(string)`**
  - Detailed LaTeX and math notation normalization:
    - Removes linebreaks, inverse spaces
    - Replaces tfrac/dfrac → frac
    - Removes \left, \right, ^{\circ}
    - Fixes fractions: `\frac1b` → `\frac{1}{b}`
    - Handles mixed numbers and negative signs

- Helper functions:
  - **`_fix_fracs(string)`**: Fixes malformed fraction notation
  - **`_fix_a_slash_b(string)`**: Converts a/b → \frac{a}{b}
  - **`_remove_right_units(string)`**: Strips unit descriptions
  - **`_fix_sqrt(string)`**: Fixes sqrt notation

##### HumanEval Execution (`he_execute.py`, `he_eval.py`)
- **`check_correctness(problem, completion, timeout, completion_id)`**
  - Executes code completion against test suite
  - Uses multiprocessing with timeout
  - Returns: dict with task_id, passed status, result

- **`unsafe_execute(problem, completion, timeout, result)`**
  - Sandboxed code execution with safety guards
  - Constructs check program: prompt + completion + test
  - Executes with timeout protection

- **`reliability_guard(maximum_memory_bytes)`**
  - Disables destructive operations (fork, kill, file removal, etc.)
  - Sets memory limits
  - Prevents interference with test environment

- **`evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)`**
  - Evaluates multiple samples per problem
  - Computes pass@k metric
  - Uses ThreadPoolExecutor for parallel execution

- **`estimate_pass_at_k(num_samples, num_correct, k)`**
  - Statistical estimator for pass@k metric
  - Formula: 1 - C(n-c, k) / C(n, k)

#### 2.4 Constants (`constants.py`)
Defines prompt templates and formatting strings.

##### Prompt Templates
- **`PROMPT`**: "Can you solve the following math problem? "
- **`BASE`**: Direct answer instruction with \boxed{}
- **`COT`**: Step-by-step reasoning instruction with \boxed{}
- **`COT_ALT`**: Alternative detailed reasoning instruction

##### Code Execution Prompts (for CRUXEVAL-style tasks)
- **`make_direct_output_prompt(s)`**: Predict output without reasoning
- **`make_cot_output_prompt(s)`**: Predict output with step-by-step reasoning
- **`make_direct_input_prompt(s)`**: Predict input from output (inverse problem)
- **`make_cot_input_prompt(s)`**: Predict input with reasoning

##### Query Templates
- **`GPQA_QUERY_TEMPLATE`**: Multiple choice question format

---

## Execution Flow

### Main Experiment Flow (`power_samp.py` main)
1. **Setup**
   - Parse command-line arguments (model, temperature, dataset, mcmc_steps)
   - Load model and tokenizer
   - Load dataset (MATH/AIME)
   
2. **Per Problem**
   - Format prompt with `format_prompt()`
   - Tokenize input
   - Generate 3 completions:
     - **Naive Temperature**: `hf_model.generate()` with temperature
     - **Standard**: `hf_model.generate()` with do_sample=True
     - **MCMC Temperature**: `mcmc_temp()` with MCMC refinement
   
3. **Post-processing**
   - Decode token IDs to text
   - Parse answers with `parse_answer()`
   - Store results in DataFrame
   - Save to CSV

### Likelihood Analysis Flow (`likelihoods.py` main)
1. Load results from multiple CSV files
2. Extract sequences (MCMC, STD, GRPO)
3. Compute log-likelihoods with `log_probs()`
4. Generate comparative histograms

---

## Key Algorithms

### Metropolis-Hastings MCMC (used in both modules)
**Purpose**: Refine samples to better match target distribution

**Algorithm**:
```
for each MCMC step:
    1. Choose random position idx in generated sequence
    2. Propose: regenerate from idx to end using base sampler
    3. Compute acceptance ratio:
       log_r = log_target(proposed) + log_proposal(current)
             - log_target(current) - log_proposal(proposed)
    4. Accept with probability min(1, exp(log_r))
    5. If accepted: replace current with proposed
```

**Variants**:
- `compositional_sampler`: Uses `naive_composition` as proposal
- `mcmc_temp`: Uses `naive_temp` as proposal
- `max_swap`: Greedy variant (accept if log_r > 0)

---

## Dependencies Hierarchy

```
toy_composition.py
├── random (sampling)
├── math (exp function)
├── numpy (log, random)
└── functools (partial)

llm_experiments/power_samp.py
├── torch (model inference, tensor ops)
├── transformers (AutoModel, AutoTokenizer, generate)
├── datasets (load_dataset)
├── pandas (DataFrame for results)
├── numpy (random, exp for MH)
└── constants.py (prompt templates)

llm_experiments/likelihoods.py
├── torch (model inference)
├── transformers (model loading)
├── pandas (data loading)
├── matplotlib (visualization)
├── grader_utils.math_grader (grading)
└── constants.py

llm_experiments/grader_utils/math_grader.py
├── sympy (symbolic math)
├── pylatexenc (LaTeX parsing)
└── grader_utils.math_normalize

llm_experiments/grader_utils/he_execute.py
├── multiprocessing (sandboxed execution)
├── signal (timeout handling)
└── tempfile (temporary directories)
```

---

## Function Call Graphs

### Compositional Sampling
```
compositional_sampler()
├── naive_composition()
│   ├── set_product()
│   ├── normalize()
│   └── [sample from distribution]
└── [MCMC loop]
    └── naive_composition() [for proposals]
```

### MCMC Temperature Sampling
```
mcmc_temp()
├── naive_temp()
│   ├── model.generate()
│   └── [compute log probs]
└── [MCMC loop]
    └── naive_temp() [for proposals]
```

### Answer Grading
```
grade_answer()
├── math_normalize.normalize_answer()
│   └── _strip_string()
│       ├── _fix_fracs()
│       ├── _fix_sqrt()
│       ├── _fix_a_slash_b()
│       └── _remove_right_units()
├── _normalize()
│   └── _parse_latex()
├── split_tuple()
└── are_equal_under_sympy()
    └── _sympy_parse()
```

---

## Configuration & Hyperparameters

### Toy Composition
- `T`: Vocabulary size parameter (default: 20)
- `seq_len`: Sequence length (default: 2*T = 40)
- `mcmc_steps`: MCMC iterations per token (default: 20)
- `N`: Number of sequences to generate (default: 20)

### LLM Experiments
- `temperature`: Temperature for scaling (e.g., 0.25, 0.5)
- `mcmc_steps`: MCMC refinements per block (e.g., 10)
- `max_new_tokens`: Maximum generation length (3072)
- `block_num`: Number of generation blocks (16)
- `jump_size`: Tokens per block = max_new_tokens / block_num

### Supported Models
- `qwen`: Qwen/Qwen2.5-7B
- `qwen_math`: Qwen/Qwen2.5-Math-7B
- `qwen_grpo`: GRPO-trained variant
- `phi`: Microsoft Phi-3.5-mini-instruct
- `tulu`: Llama-3.1-Tulu-3-8B-DPO
- `gptoss_high`: OpenAI GPT-OSS-20B

---

## Output Formats

### CSV Output (power_samp.py)
Columns:
- `question`: Problem statement
- `correct_answer`: Ground truth
- `naive_completion`: Temperature-scaled generation
- `naive_answer`: Parsed answer from naive
- `std_completion`: Standard generation
- `std_answer`: Parsed answer from std
- `mcmc_completion`: MCMC-refined generation
- `mcmc_answer`: Parsed answer from MCMC

### Visualization Output (likelihoods.py)
- `hist_loglikelihoods.png`: Distribution of log-likelihoods
- `hist_confidence.png`: Distribution of confidence scores

---