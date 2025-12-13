# Standard Libraries

# Math Libraries
import numpy as np

# Custom Libraries
from pita.inference.LLM_backend import AutoregressiveSampler

# SMC Class
class SequentialMonteCarlo_Params:
    def __init__(
        self, 
        num_particles: int = 10, 
        tokens_per_step: int = 5, 
        stop_on_eos: bool = True
    ):
        self.num_particles = num_particles
        self.tokens_per_step = tokens_per_step
        self.stop_on_eos = stop_on_eos

# Enable SMC Sampling Function
def enable_smc_sampling(
    sampler: AutoregressiveSampler, 
    num_particles: int = 10, 
    tokens_per_step: int = 5, 
    stop_on_eos: bool = True
) -> None:
    # Check if the sampler is initialized
    if(sampler is None):
        raise ValueError("Sampler must be initialized before enabling SMC sampling.")
    
    sampler.smc_sampling_params = SequentialMonteCarlo_Params(
        num_particles=num_particles,
        tokens_per_step=tokens_per_step,
        stop_on_eos=stop_on_eos
    )

def SequentialMonteCarlo(
    sampler: AutoregressiveSampler, 
    prompt: str
) -> str:

        # Initialize particles with the prompt 
        particles = [prompt for _ in range(sampler.smc_sampling_params.num_particles)]

        # cumulative log weights for each particle
        cum_logit_weights = np.zeros(sampler.smc_sampling_params.num_particles, dtype=float)

        # finished flags per particle
        finished = np.zeros(sampler.smc_sampling_params.num_particles, dtype=bool)
        eos_id = getattr(sampler.tokenizer, "eos_token_id", None)

        total_tokens_generated = 0

        while total_tokens_generated < sampler.sampling_params.max_tokens:
            # determine how many tokens to generate in this step
            max_new_tokens = min(sampler.smc_sampling_params.tokens_per_step, sampler.sampling_params.max_tokens - total_tokens_generated)
            print("Max New Tokens this step:", max_new_tokens)
            for i in range(sampler.smc_sampling_params.num_particles):
                if finished[i]:
                    continue # skip sampling for finished particles 
                context = particles[i]
                tokens, token_logits, _ = sampler.sample(context, max_new_tokens) # get tokens and their logprobs

                # tokens expected as list of token ids
                # If EOS detection enabled, truncate and mark finished
                if sampler.smc_sampling_params.stop_on_eos and eos_id is not None:
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
            
            # Normalize cumulative logit weights to prevent numerical issues
            # TO DO: Do we need to normalize over token count to prevent bias towards longer sequences?
            max_cum_logit_weight = np.max(cum_logit_weights)
            weight = np.exp(cum_logit_weights - max_cum_logit_weight)
            normalized_weights = weight / np.sum(weight)

            # Resample indices according to normalized weights (multinomial resampling)
            # Indices creates an 1 x num_particles array of selected particle indices
            indices = np.random.choice(
                sampler.smc_sampling_params.num_particles, 
                size=sampler.smc_sampling_params.num_particles, 
                p=normalized_weights
            )
            # Set the new particle array according to resampled indices
            particles = [particles[idx] for idx in indices]
            # Assign the cum_logit_weights to the corresponding cum_log_weights of the resampled particles
            cum_logit_weights = cum_logit_weights[indices]
            # Update finished flags
            finished = finished[indices]

            # if all particles finished, we can stop early
            if finished.all():
                break

            total_tokens_generated += max_new_tokens

        # Greedy select the best particle by cumulative logit weight and return decoded text
        # Find the index of the particle with the highest cumulative logit weight
        best_idx = int(np.argmax(cum_logit_weights))

        # Decode the best particle token IDs to text and return it to the user
        best_particle_response = particles[best_idx]

        return best_particle_response