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
    Sequential Monte Carlo (SMC) is a multi-particle sampling method that is uses a probability metric to iteratively update a set of particles.

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
            step_scores (list[list[float]]): The stored step scores.

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
        particle_scores: list[float],
        finished: list[bool]
    ) -> list[int]:
        """
        Given a list of particle scores (particle_score), return a list of the new particles to use based off the softmax of the particle scores and multinomial sampling.
        Skip any particles that have finished.

        Args:
            particle_scores (list[float]): The list of particle scores.
            finished (list[bool]): The list of finished flags.

        Returns:
            list[int]: A list that with each element being the new index of the particle to use.
        """
        # Find the indices of the unfinished particles
        unfinished_indices = np.where(np.array(finished) == False)[0]
        
        # If all particles are finished, return the current particles
        if len(unfinished_indices) == 0:
            return list(range(self.num_particles))

        # Find the normalization constant of the particle scores 
        particle_score_exp = np.exp(np.array(particle_scores)[unfinished_indices])
        
        # Find the normalization constant of the particle scores
        particle_score_normalization_constant = np.sum(particle_score_exp)
        
        # Normalize the particle scores
        normalized_scores = particle_score_exp / particle_score_normalization_constant

        # Choose the new particle with multinomial sampling skipping any finished particles
        new_particles = np.random.choice(unfinished_indices, size=self.num_particles, p=normalized_scores)
        
        # Make sure the finished particles are propagated forward
        for i in range(self.num_particles):
            if(finished[i] == True):
                new_particles[i] = i
        

        return new_particles.tolist()
    def update_particles(
        self,
        new_particles: list[int],
        outputs: list[Output],
        finished: list[bool],
        token_metric_scores: list[list[float]],
        step_scores: list[list[float]]
    ) -> None:
        """
        Update the particles based on the newly SMC sampled particles by updating the outputs, token_metric_scores, and step_scores 

        Args:
            new_particles (list[int]): The list of indices of the new particles to use.
            outputs (list[Output]): The current list of outputs to be updated.
            finished (list[bool]): The current list of finished flags to be updated.
            token_metric_scores (list[float]): The current list of token metric scores to be updated.
            step_scores (list[float]): The current list of step scores to be updated.

        """
        # Save the particles that are will be carried forward
        # Find the unique indices in new_particles avoiding any finished particles
        unique_indices = np.unique(new_particles)
        for i in range(self.num_particles):
            if(finished[i] == True):
                unique_indices = unique_indices[unique_indices != i]

        # Save the outputs, finished, token_metric_scores, and step_scores for the unique indices in dictionaries
        saved_outputs = {i: outputs[i] for i in unique_indices}
        saved_finished = {i: finished[i] for i in unique_indices}
        saved_token_metric_scores = {i: token_metric_scores[i] for i in unique_indices}
        saved_step_scores = {i: step_scores[i] for i in unique_indices}

        for i in range(self.num_particles):
            source_idx = new_particles[i]
            # Check to see if the particle is different
            if(source_idx != i):
                # Set the current particle to the new saved particle
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
        if(self.token_sampling_method == "standard"):
            token_sampling = sampler.sample 
        elif(self.token_sampling_method == "token_sample"):
            token_sampling = sampler.token_sample
        else:
            print("Warning: Invalid token sampling method. Using standard token sampling method.")
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

        # Create a list of the token metric probabilites for each particle and store them in a list
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
                token_metric_scores[particle].extend(calc_token_metric(sample_output, sampler, self.token_metric))

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
