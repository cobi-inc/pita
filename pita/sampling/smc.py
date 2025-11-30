# Standard Libraries
import os
import random

# Math Libraries
import numpy as np

# Custom Libraries
from pita.inference.autoregressive_sampler_backend import AutoregressiveSampler

def SequentialMonteCarlo(
    sampler: AutoregressiveSampler, 
    prompt: str, 
    num_particles: int = None, 
    tokens_per_step: int = None, 
    max_tokens: int = None, 
    stop_on_eos: bool = True
) -> str:

        # Initialize particles with the prompt 
        particles = [prompt for _ in range(num_particles)]

        # cumulative log weights for each particle
        cum_logit_weights = np.zeros(num_particles, dtype=float)

        # finished flags per particle
        finished = np.zeros(num_particles, dtype=bool)
        eos_id = getattr(sampler.tokenizer, "eos_token_id", None)

        total_tokens_generated = 0

        while total_tokens_generated < max_tokens:
            # determine how many tokens to generate in this step
            max_new_tokens = min(tokens_per_step, max_tokens - total_tokens_generated)

            for i in range(num_particles):
                if finished[i]:
                    continue # skip sampling for finished particles 
                context = particles[i]
                tokens, token_logits, _ = sampler.sample(context, max_new_tokens) # get tokens and their logprobs

                # tokens expected as list of token ids
                # If EOS detection enabled, truncate and mark finished
                if stop_on_eos and eos_id is not None:
                    if eos_id in tokens:
                        idx = tokens.index(eos_id) + 1
                        tokens = tokens[:idx]
                        finished[i] = True

                # append generated tokens 
                tokens = sampler.tokenizer.decode(tokens)
                particles[i] = context + tokens 

                # for each particle generated block find cumulative logit sum
                block_logits = float(np.sum(token_logits)) if len(token_logits) > 0 else 0.0
                cum_logit_weights[i] += block_logits # accumulate from previous block generation 

            max_cum_logit_weight = np.max(cum_logit_weights)
            weight = np.exp(cum_logit_weights - max_cum_logit_weight)
            normalized_weights = weight / np.sum(weight)

            # Resample indices according to normalized weights (multinomial resampling)
            indices = np.random.choice(num_particles, size=num_particles, p=normalized_weights)
            particles = [particles[idx] for idx in indices]
            cum_logit_weights = cum_logit_weights[indices]
            finished = finished[indices]

             # if all particles finished, we can stop early
            if finished.all():
                break

            total_tokens_generated += max_new_tokens

        # Select the best particle by cumulative log-weight and return decoded text
        best_idx = int(np.argmax(cum_log_weights))
        best_particle_ids = particles[best_idx]
        return sampler.tokenizer.decode(best_particle_ids, skip_special_tokens=True)