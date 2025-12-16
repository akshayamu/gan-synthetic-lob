import numpy as np
from scipy.spatial.distance import pdist

def diversity_score(samples):
    return {
        "variance": np.mean(np.var(samples, axis=0)),
        "pairwise_distance": pdist(samples[:200]).mean(),
        "unique_rows": len(np.unique(samples.round(4), axis=0))
    }
