import numpy as np
from pita.sampling.smc import Sequential_Monte_Carlo

class TestParticleSampling:
    def test_particle_sampling_basic(self):
        """
        Test that particle_sampling returns a list of the correct size
        when no particles are finished.
        """
        num_particles = 5
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        # Create dummy scores (all equal prob)
        particle_scores = np.zeros(num_particles)
        finished = np.zeros(num_particles, dtype=bool)
        
        new_particles = smc.particle_sampling(particle_scores, finished)
        
        # Check return type and shape
        # expecting list or array of indices
        assert hasattr(new_particles, '__len__'), "Result should be iterable"
        assert len(new_particles) == num_particles, f"Expected {num_particles} new particles"
        
        # Check values are valid indices
        for p in new_particles:
            assert isinstance(p, (int, np.integer))
            assert 0 <= p < num_particles

    def test_particle_sampling_with_finished(self):
        """
        Test particle_sampling when some particles are finished.
        This often requires careful handling of the probability array shapes.
        """
        num_particles = 4
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        # 2 particles finished
        particle_scores = np.ones(num_particles)
        finished = np.array([False, True, False, True]) # Indices 1 and 3 finished
        
        # The logic inside particle_sampling needs to handle the fact that
        # we still need to return num_particles new particles (presumably),
        # or it should select from the unfinished ones to replace?
        # The docstring says: "return a list of the new particles to use... Skip any particles that have finished."
        
        new_particles = smc.particle_sampling(particle_scores, finished)
        
        # Test to make sure the number of particles returned is the same as the number of particles
        assert len(new_particles) == num_particles

        # Test to make sure the finished particles are propagated forward
        for i in range(num_particles):
            if(finished[i]):
                assert new_particles[i] == i

        # Test to make sure that the unfished particles are not assigned to finished particles
        for i in range(num_particles):
            if(not finished[i]):
                # new_particles[i] should not point to any finished particle index (1 or 3)
                assert new_particles[i] != 1
                assert new_particles[i] != 3

    def test_particle_sampling_normalization(self):
        """
        Test that probabilities are normalized correctly before sampling.
        We can infer this by setting one score very high and checking if it's selected.
        """
        num_particles = 10
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        particle_scores = np.zeros(num_particles)
        particle_scores[0] = 100.0 # High score for particle 0
        finished = np.zeros(num_particles, dtype=bool)
        
        new_particles = smc.particle_sampling(particle_scores, finished)
        
        # With high score, particle 0 should be selected for all/most slots
        # Since logic presumably samples with replacement
        assert 0 in new_particles
        # It's probabilistic, but with 100.0 vs 0.0 in exp space, it's deterministic for all practical purposes
        assert np.all(np.array(new_particles) == 0)

    def test_particle_sampling_normalization_with_finished(self):
        """
        Test that probabilities are normalized correctly before sampling.
        We can infer this by setting one score very high and checking if it's selected.
        """
        num_particles = 10
        smc = Sequential_Monte_Carlo(num_particles=num_particles)
        
        particle_scores = np.zeros(num_particles)
        particle_scores[1] = 100.0 # High score for particle 1
        finished = [True, False, False, False, False, False, False, False, True, False]
        
        new_particles = smc.particle_sampling(particle_scores, finished)
        # With high score, particle 1 should be selected for all unfinished slots
        # Since logic presumably samples with replacement
        assert new_particles[0] == 0
        assert new_particles[8] == 8
        # It's probabilistic, but with 100.0 vs 0.0 in exp space, it's deterministic for all practical purposes
        # Check that every particle is either finished or is particle 1
        for i in range(num_particles):
            if(finished[i]):
                assert new_particles[i] == i
            else:
                assert new_particles[i] != 0
                assert new_particles[i] != 8
                assert new_particles[i] == 1