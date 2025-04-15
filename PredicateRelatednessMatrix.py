import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
from tqdm import tqdm
import math
from itertools import combinations
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.linalg import block_diag
from concurrent.futures import ThreadPoolExecutor
import os
TF_FILE="TF.pkl"
IDF_FILE="IDF.pkl"
REL_MATRIX_FILE="relatedness_matrix.pkl"
GRAPH_PICKLE_FILE = 'graph.pkl'


class PredicateRelatednessMatrix:
    def __init__(self, triples_df: pd.DataFrame):
        self.triples_df = triples_df
        self.predicates = sorted(triples_df['p'].unique())
        self.pred_to_idx = {p: i for i, p in enumerate(self.predicates)}
        self.num_predicates = len(self.predicates)
        # Group triples by predicate for faster processing
        self.groups = triples_df.groupby('p')
        self.n_preds = len(self.predicates)
        

    def compute_tf_parallel(self,groups, split_preds, rows, cols):
        """
        Compute the TF values for the given pairs of predicates using multithreading.
        """
        def compute_tf(i, j):
            group1 = groups.get_group(split_preds[i])
            group2 = groups.get_group(split_preds[j])
            c1=len(set(group1['s']).intersection(set(group2['s'])))
            c2=len(set(group1['o']).intersection(set(group2['o'])))
            return np.log(1 +(c1+c2) )
        
        with ThreadPoolExecutor(max_workers=15) as executor:
            with tqdm(total=len(rows)) as pbar:
                # Wrap the compute_tf function and track progress
                def compute_with_progress(pair):
                    result = compute_tf(pair[0], pair[1])
                    pbar.update(1)
                    return result
                # Execute the computation
                tf_values = list(executor.map(compute_with_progress, zip(rows, cols)))
        return tf_values
       

    def compute_matrix(self) -> List[np.ndarray]:
        """Compute relatedness matrix in splits using TF-IDF and cosine similarity."""
        rel_matrix = np.zeros((self.num_predicates, self.num_predicates))
        upper_tri_indices = np.triu_indices(self.num_predicates)
        rows, cols = upper_tri_indices
        split_preds = self.predicates
        n_preds_split=self.num_predicates
        if os.path.exists(TF_FILE)  :
            with open("TF.pkl", 'rb') as f:
                tf = pickle.load(f)
        else:
            print("Starting computing TF.....")
            # Compute TF (Term Frequency) matrix
            tf=self.compute_tf_parallel(self.groups, split_preds, rows, cols)
            with open(TF_FILE, 'wb') as f:
                pickle.dump(tf, f)
        print("Finished...TF....")
        
        # Populate the TF matrix
        pos = [(i, j) for i, j in zip(rows, cols)]
        if pos:
           rows, cols = zip(*pos)
           rel_matrix[rows, cols] = tf
        
        if os.path.exists(IDF_FILE)  :
            with open("IDF.pkl", 'rb') as f:
                idf_values = pickle.load(f)
        else:
            print("Starting computing IDF.....")
            idf_values = {
            i: np.log(self.n_preds / np.count_nonzero(rel_matrix[i]))
            for i in range(self.num_predicates)}
            with open(IDF_FILE, 'wb') as f:
                pickle.dump(idf_values, f)
        print("Finished...IDF....")
       
        # Apply TF-IDF transformation
        tf_idf = [
            rel_matrix[i][j] * idf_values.get(i, 0)
            for i in range(self.num_predicates)
            for j in range(self.num_predicates)
        ]
        print(tf_idf)

        pos = [(i, j) for i in range(n_preds_split) for j in range(n_preds_split)]
        if pos:
            rows, cols = zip(*pos)
            rel_matrix[rows, cols] = tf_idf
        
        # Compute Cosine Similarity
        similarity_matrix = np.zeros((n_preds_split, n_preds_split))
        for i in range(n_preds_split):
            for j in range(n_preds_split):
                similarity_matrix[i, j] = cosine_similarity(
                    rel_matrix[i].reshape(1, -1),
                    rel_matrix[j].reshape(1, -1)
                )[0][0]
        

        similarity_matrix = np.maximum(similarity_matrix, similarity_matrix.T)
        similarity_matrix = similarity_matrix +0.00001
        return similarity_matrix

 
    def save_matrix(self, matrix: np.ndarray, filepath: str):
        """Save the computed matrix and metadata."""
        data = {
            'matrix': matrix,
            'predicates': self.predicates,
            'pred_to_idx': self.pred_to_idx
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load_matrix(cls, filepath: str) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        """Load a previously saved matrix and metadata."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data['matrix'], data['predicates'], data['pred_to_idx']

def compute(graph, output_file: str):
    graph = graph[1:]  # Skip the header
     # Create DataFrame from graph
    df = pd.DataFrame(graph, columns=['s', 'p', 'o'])  # Explicit column name
    processor = PredicateRelatednessMatrix(df)
    final_matrix = processor.compute_matrix()
    processor.save_matrix(final_matrix,output_file)
    return final_matrix, processor.predicates,processor.pred_to_idx

def load_graph_from_pickle(file_path):
    """Load Python object from a Pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
# Example usage
if __name__ == "__main__":    
    if os.path.exists(REL_MATRIX_FILE)  :
        with open(REL_MATRIX_FILE, 'rb') as f:
            data=pickle.load(f)
            relatedness_matrix=data['matrix']
            predicates=data['predicates']
            predicates_to_idx=data['predicates_to_idx']
    else:
        print("Computing relatedness matrix....")
        graph_data = load_graph_from_pickle(GRAPH_PICKLE_FILE)
        graph=graph_data[0]
        matrix, predicates,pred_to_idx = compute(graph, output_file="relatedness_matrix.pkl")
    
    print((matrix.shape),len(predicates),len(pred_to_idx))
