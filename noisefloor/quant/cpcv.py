"""
Combinatorial Purged Cross-Validation (CPCV).
"""
import itertools
import numpy as np

class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV)
    Generates train/test indices while purging observations that leak 
    information and embargoing after training to prevent look-ahead bias.
    """
    def __init__(
        self, 
        n_splits: int = 6, 
        n_test_splits: int = 2, 
        purge_gap: int = 0, 
        embargo_gap: int = 0
    ):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_gap = embargo_gap
        
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Determine the boundaries for the groups
        split_indices = np.array_split(indices, self.n_splits)
        
        # Generate all combinations of test groups
        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        
        for test_groups in combinations:
            test_indices = []
            for g in test_groups:
                test_indices.extend(split_indices[g])
            test_indices = np.array(test_indices)
            
            # Find training groups
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]
            
            train_indices = []
            for g in train_groups:
                group_indices = split_indices[g]
                
                # Apply purging and embargoing logic
                valid_indices = group_indices.copy()
                
                # For each test group, apply purge and embargo
                for test_g in test_groups:
                    test_min = split_indices[test_g][0]
                    test_max = split_indices[test_g][-1]
                    
                    # Purging: remove training data immediately preceding test data
                    if g < test_g:
                        purge_end = test_min
                        purge_start = test_min - self.purge_gap
                        valid_indices = valid_indices[(valid_indices < purge_start) | (valid_indices >= purge_end)]
                    
                    # Embargo: remove training data immediately following test data
                    if g > test_g:
                        embargo_start = test_max
                        embargo_end = test_max + self.embargo_gap
                        valid_indices = valid_indices[(valid_indices <= embargo_start) | (valid_indices > embargo_end)]
                        
                train_indices.extend(valid_indices)
                
            train_indices = np.array(train_indices)
            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        return len(combinations)
