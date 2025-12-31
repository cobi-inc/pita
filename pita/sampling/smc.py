# Math Libraries
import numpy as np
import math
import copy

# Custom Libraries
from pita.inference.LLM_backend import AutoregressiveSampler, Output
from pita.sampling.token_metrics import calc_token_metric

# SMC Class
class Sequential_Monte_Carlo:
    """
    Sequential Monte Carlo (SMC) is a multi-particle sampling method that uses a probability metric to iteratively update a set of particles.

    Args:
        num_particles (int): The number of particles to use.
        tokens_per_step (int): The number of tokens to generate per step.
        stop_on_eos (bool): Whether to stop sampling when the end of the sequence is reached.
        token_metric (str): The probability metric to use.
        aggregation (str): The aggregation method to use. Can be the last, minimum, product, or model_aggregate.
        token_sampling_method (str): The token sampling method to use. By default, the standard token sampling method is used. However, token_sample can be used instead. 
    """
    def __init__(
        self, 
        num_particles: int = 10, 
        tokens_per_step: int = 5, 
        stop_on_eos: bool = True,
        token_metric: str = "logprobs",
        aggregation: str = "last",
        token_sampling_method: str = "standard"
    ):
        self.num_particles = num_particles
        self.tokens_per_step = tokens_per_step
        self.stop_on_eos = stop_on_eos
        self.token_metric = token_metric
        self.aggregation = aggregation
        self.token_sampling_method = token_sampling_method

    def score_update(
        self, 
        token_values: list[float],
        token_count: int,
        step_scores: list[float]
    ) -> float:
        """
        Update the particle score and stored step score for the new token scores.

        Args:
            token_values (list[float]): All of the token values so far. Could be logprobs, power_distribution, or entropy
            token_count (int): The number of tokens to use.
            step_scores (list[float]): The stored step scores.

        Returns:
            float: The new particle score.
        """
        if(self.token_metric == "logprobs" or self.token_metric == "power_distribution"):
            if(self.aggregation == "last"):
                step_scores.append(sum(token_values[-token_count:])/token_count)
                return step_scores[-1]
            elif(self.aggregation == "minimum"):
                step_scores.append(sum(token_values[-token_count:])/token_count)
                return min(step_scores)
            elif(self.aggregation == "product"):
                step_scores.append(sum(token_values[-token_count:])/token_count)
                return -1*abs(math.prod(step_scores))
            elif(self.aggregation == "model_aggregate"):
                step_scores.append(sum(token_values[-token_count:])/token_count)
                return sum(token_values)/len(token_values)
            else:
                raise ValueError(f"Invalid aggregation method: {self.aggregation}")
        elif(self.token_metric == "entropy"):
            # As a high entropy is less desirable, we negate the value or take the maximum value before negating
            if(self.aggregation == "last"):
                step_scores.append(-1 * sum(token_values[-token_count:])/token_count)
                return step_scores[-1]
            elif(self.aggregation == "minimum"):
                step_scores.append(-1 * sum(token_values[-token_count:])/token_count)
                return min(step_scores)
            elif(self.aggregation == "product"):
                step_scores.append(-1 * sum(token_values[-token_count:])/token_count)
                return -1 * abs(math.prod(step_scores))
            elif(self.aggregation == "model_aggregate"):
                step_scores.append(-1 * sum(token_values[-token_count:])/token_count)
                return -1 * sum(token_values)/len(token_values)
            else:
                raise ValueError(f"Invalid aggregation method: {self.aggregation}")
        else:
            raise ValueError(f"Invalid token metric: {self.token_metric}")
    
    def particle_sampling(
        self,
        particle_scores,
        finished
    )-> list[int]:
        """
        Given a list of particle scores (particle_score), return a list of the new particles to use based off the softmax of the particle scores and multinomial sampling.
        Skip any particles that have finished.

        Args:
            particle_scores (list[float]): The list of particle scores.
            finished (list[bool]): The list of finished flags.

        Returns:
            list[int]: A list with each element being the new index of the particle to use.
        """
        # Find the indices of the unfinished particles
        unfinished_indices = np.where(np.logical_not(finished))[0]
        
        # If all particles are finished, return the current particles
        if len(unfinished_indices) == 0:
            return list(range(self.num_particles))

        # Exponentiate the scores of unfinished particles (softmax numerator)
        particle_score_exp = np.exp(np.array(particle_scores)[unfinished_indices])
        
        # Calculate the sum of exponentiated scores for normalization
        particle_score_normalization_constant = np.sum(particle_score_exp)
        
        # Normalize the particle scores
        normalized_scores = particle_score_exp / particle_score_normalization_constant

        # Choose the new particle with multinomial sampling skipping any finished particles
        new_particles = np.random.choice(unfinished_indices, size=self.num_particles, p=normalized_scores)
        
        # Make sure the finished particles are propagated forward
        for i in range(self.num_particles):
            if finished[i]:
                new_particles[i] = i
        

        return new_particles.tolist()

    def update_particles(
        self,
        new_particles: list[int],
        outputs,
        finished,
        token_metric_scores,
        step_scores,
        
    ) -> None:
        """
        Update the particles based on the newly SMC sampled particles by updating the outputs, token_metric_scores, and step_scores 

        Args:
            new_particles (list[int]): The list of indices of the new particles to use.
            outputs (list[Output]): The current list of outputs to be updated.
            finished (list[bool]): The current list of finished flags to be updated.
            token_metric_scores (list[list[float]]): The current list of token metric scores to be updated for each particle.
            step_scores (list[list[float]]): The current list of step scores to be updated for each particle.

        """
        # Save the states that will be carried forward
        # Only save states for particles that are actually used in new_particles
        saved_outputs = {}
        saved_finished = {}
        saved_token_metric_scores = {}
        saved_step_scores = {}
        
        for source_idx in new_particles:
            if source_idx not in saved_outputs:
                saved_outputs[source_idx] = outputs[source_idx]
                saved_finished[source_idx] = finished[source_idx]
                saved_token_metric_scores[source_idx] = token_metric_scores[source_idx]
                saved_step_scores[source_idx] = step_scores[source_idx]

        for i in range(self.num_particles):
            source_idx = new_particles[i]
            # Check to see if the particle is different
            if source_idx != i:
                # Copy the saved particle to the current particle
                # Use deepcopy to avoid shared mutable state
                outputs[i] = copy.deepcopy(saved_outputs[source_idx])
                finished[i] = saved_finished[source_idx]
                token_metric_scores[i] = copy.deepcopy(saved_token_metric_scores[source_idx])
                step_scores[i] = copy.deepcopy(saved_step_scores[source_idx])

    # TODO add support for a PRM token metric
    def sample(
        self,
        sampler: AutoregressiveSampler,
        prompt: str
    ) -> Output:
        """
        Samples using SMC and its parameters.

        Args:
            sampler (AutoregressiveSampler): The sampler object.
            prompt (str): The prompt to sample from.
        Returns:
            Output: Standard output object for the PITA library.
        """
        # Check the token sampling method
        if self.token_sampling_method == "standard":
            token_sampling = sampler.sample 
        elif self.token_sampling_method == "token_sample":
            token_sampling = sampler.token_sample
        else:
            import warnings
            warnings.warn("Invalid token sampling method. Using standard token sampling method.")
            token_sampling = sampler.sample

        # Save the total number of tokens to generate
        total_tokens = sampler.sampling_params.max_tokens
        # Set the number of tokens to generate per step
        sampler.sampling_params.max_tokens = self.tokens_per_step

        # SMC Steps
        smc_steps = total_tokens // self.tokens_per_step

        # Create Output objects for each particle and store them in a list
        # Initialize each particle with an empty Output object configured for appending
        outputs = [
            Output(
                tokens=[], 
                top_k_logits=[], 
                top_k_logprobs=[],
                unprocessed_log_normalization_constant=[],
                temp_processed_log_normalization_constant=[],
                entropy=[]
            ) 
            for _ in range(self.num_particles)
        ]

        # Create a list of the token metric probabilities for each particle and store them in a list
        token_metric_scores = [[] for i in range(self.num_particles)]
        step_scores = [[] for i in range(self.num_particles)]
        # Create a list of the current particle probability
        particle_scores = [0 for i in range(self.num_particles)]

        # Create a list to track if a particle has finished
        finished = [False for i in range(self.num_particles)]
        
        # Find the EOS token ID
        eos_id = sampler.tokenizer.eos_token_id
        
        # Iterate through the SMC
        for step in range(smc_steps):
            for particle in range(self.num_particles):
                # If the particle has finished, skip it
                if finished[particle]:
                    continue
                
                # Sample the next token
                sample_output = token_sampling(prompt + sampler.tokenizer.decode(outputs[particle].tokens, skip_special_tokens=True))

                # Append the sample output to the particle's output
                outputs[particle].append(sample_output)

                # Calculate the token metric probabilities
                token_metric_scores[particle].extend(
                    np.ravel(calc_token_metric(sample_output, sampler, self.token_metric)).tolist()
                )

                # Calculate the current particle probability
                particle_scores[particle] = self.score_update(token_metric_scores[particle], self.tokens_per_step, step_scores[particle]) 

                # Check if the particle has finished 
                if self.stop_on_eos and eos_id in sample_output.tokens:
                    finished[particle] = True

            # Find the new list of particles to use
            new_particles = self.particle_sampling(particle_scores, finished)
            
            # Check if all particles are finished
            if np.all(finished):
                break

            # Update the particles if not finished
            self.update_particles(new_particles, outputs, finished, token_metric_scores, step_scores)

        # Greedily select the best particle
        best_particle = np.argmax(particle_scores)

        # Restore the sampling parameters
        sampler.sampling_params.max_tokens = total_tokens

        # Return the best particle
        return outputs[best_particle]

# TODO Remove this function as it will be incorporated into the LLM_backend.py
# Enable SMC Sampling Function
# Take in the default parameters for SMC sampling and set them in the sampler object
def enable_smc_sampling(
    sampler: AutoregressiveSampler, 
    num_particles: int = 10, 
    tokens_per_step: int = 5, 
    stop_on_eos: bool = True,
    token_metric: str = "logprobs",
    aggregation: str = "last"
) -> None:
    # Check if the sampler is initialized
    if(sampler is None):
        raise ValueError("Sampler must be initialized before enabling SMC sampling.")
    
    sampler.smc_sampling_params = Sequential_Monte_Carlo_Params(
        num_particles=num_particles,
        tokens_per_step=tokens_per_step,    
        stop_on_eos=stop_on_eos
    )

# TODO Determine if this function is really needed anymore
# Use a standard temperature affected log probability metric to perform SMC sampling
# Input Variables
# sampler: The sampler object
# tokens: The list of tokens generated by the sampler
# top_k_logits: The list of logits for the top k tokens
# top_k_logprobs: The list of log probabilities for the top k tokens
# unprocessed_normalization_constant: The unprocessed normalization constant from the sample() function
# temp_processed_normalization_constant: The temperature processed normalization constant sample() function
# Returns: The list of log probabilities for each token in the generated sequence
def standard_log_probability_metric(
    sampler: AutoregressiveSampler,
    tokens: list[int], 
    top_k_logits: list[float], 
    top_k_logprobs: list[float], 
    unprocessed_normalization_constant: list[float], 
    temp_processed_normalization_constant: list[float]
) -> list[float]:
    # Want to return a list of log probabilities for each token in the generated sequence
    # Check if the top_k_logprobs is None
    if(top_k_logprobs is None and top_k_logits is not None):
        logits_list = process_top_k_probs(top_k_logits, tokens)
        probability_list = low_temp_logprobs(logits_list, temp_processed_normalization_constant, sampler.sampling_params.temperature)
    elif(top_k_logprobs is not None):
        probability_list = process_top_k_probs(top_k_logprobs, tokens)
    else:
        raise ValueError("top_k_logprobs and top_k_logits must both not be None.")
    
    return probability_list

# TODO Determine if this function is really needed anymore
# Use the power sampling log probability metric to perform the SMC sampling
# Input Variables
# sampler: The sampler object
# tokens: The list of tokens generated by the sampler
# top_k_logits: The list of logits for the top k tokens
# top_k_logprobs: The list of log probabilities for the top k tokens
# unprocessed_normalization_constant: The unprocessed normalization constant from the sample() function
# temp_processed_normalization_constant: The temperature processed normalization constant sample() function
# Returns: The list of log probabilities for each token in the generated sequence
def power_sampling_logprobability_metric(
    sampler: AutoregressiveSampler,
    tokens: list[int], 
    top_k_logits: list[float], 
    top_k_logprobs: list[float], 
    unprocessed_normalization_constant: list[float], 
    temp_processed_normalization_constant: list[float],
) -> list[float]:
    # Check if you have the logits
    if(top_k_logits is not None):
        # Find the power sampling log probabilities from the logits and unprocessed normalization constant
        logits_list = process_top_k_probs(top_k_logits, tokens)
        probability_list = power_sampling_logprobs(logits_list, unprocessed_normalization_constant, sampler.sampling_params.temperature)
    else: 
        raise ValueError("top_k_logits must be provided if the Power Sampled Log Probability Metric is used.")
    
    return probability_list

# Use negative entropy as a comparison metric for the SMC sampling

# TODO incorporate this function into the Sequential_Monte_Carlo class
# Perform SMC Sampling Function based on the log probabilities of the tokens generated
def sequential_monte_carlo(
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
                tokens, top_k_logits, top_k_logprobs, unprocessed_normalization_constant, temp_processed_normalization_constant = sampler.sample(context, max_new_tokens) # get tokens and their logprobs

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

                # calculate the comparison metric
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