from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
import numpy as np


class FeatureTransformer:
    def __init__(self, env, n_components=500):
        # Sample 10,000 observations from the environment's observation space
        observation_examples = np.array([env.observation_space.sample() for _ in range(10000)])

        # Initialize and fit a StandardScaler to normalize observations
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        # Initialize a FeatureUnion of RBFSamplers with different gamma values to capture
        # features at different scales. This enriches the feature set for better model performance.
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=n_components)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=n_components)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=n_components)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=n_components))
        ])

        # Transform the example observations to get the new feature dimensions
        example_features = featurizer.fit_transform(scaler.transform(observation_examples))

        # Store the dimensions of the transformed features, the scaler, and the featurizer
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        # print "observations:", observations
        scaled = self.scaler.transform(observations)
        # assert(len(scaled.shape) == 2)
        return self.featurizer.transform(scaled)
