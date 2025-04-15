import os
import gzip
import tarfile
import pickle
import numpy as np
from collections import deque, defaultdict
from scipy.sparse import coo_matrix
import pandas as pd
from PredicateRelatednessMatrix import compute
import logging
import string


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
class ExtractRelevantSubGraph:
    def __init__(self, dataset_name, max_depth=1, min_k=1, k_fraction=0.6):
        self.dataset_name = dataset_name
        self.max_depth = max_depth
        self.min_k = min_k
        self.k_fraction = k_fraction

        # Set base paths
        self.base_path = "../"  # Adjust if needed
        self.data_path = os.path.join(self.base_path, dataset_name) + "/"

        # File paths
        self.graph_pickle = f'{self.data_path}graph.pkl'
        self.relatedness_matrix_file = f'{self.data_path}relatedness_matrix.pkl'


        # Load data
        graph, adjacency_list, entity_dict, entity_reverse_dict,relation_dict = self.load_or_process_data()
        relatedness_matrix, predicates, pred_to_idx = self.compute_or_load_relatedness_matrix(graph)

        # Store attributes needed for extraction
        self.adjacency_list = adjacency_list
        self.pred_to_idx = pred_to_idx
        self.relation_similarity_matrix = relatedness_matrix
        self.entity_dict = entity_dict
        self.entity_reverse_dict=entity_reverse_dict
        self.relation_dict = relation_dict
        self.graph = graph

    def load_or_process_data(self):
        if os.path.exists(self.graph_pickle):
            print("Loading processed data...")
            graph_loaded = self.load_from_pickle(self.graph_pickle)
            graph = graph_loaded[0]
            adjacency_list = graph_loaded[1]
            entity_dict = self.load_from_pickle(self.entity_dict_pickle)
            entity_reverse_dict = self.load_from_pickle(self.entity_dict_reverse_pickle)
            relation_dict = self.load_from_pickle(self.relation_dict_pickle)
        else:
            print("Reading raw data and computing graph and dictionaries...")
            graph = self.read_graph_from_gz(self.gz_file_path)
            adjacency_list = self.preprocess_graph(graph)
            self.save_to_pickle([graph, adjacency_list], self.graph_pickle)

            entity_dict, relation_dict = self.read_dictionaries_from_tar_gz(
                self.tar_gz_file_path, self.entity_dict_name, self.relation_dict_name
            )
            entity_reverse_dict=self.build_reverse_mapping_multiple(entity_dict)

            self.save_to_pickle(entity_dict, self.entity_dict_pickle)
            self.save_to_pickle(entity_reverse_dict, self.entity_dict_reverse_pickle)

            self.save_to_pickle(relation_dict, self.relation_dict_pickle)

        return graph, adjacency_list, entity_dict, entity_reverse_dict,relation_dict

    def remove_punctuation(self,text):
        """
        Removes punctuation from a given string.

        Args:
            text (str): The input string.

        Returns:
            str: The string without punctuation.
        """
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def build_reverse_mapping_multiple(self,entity_dict, case_sensitive=True, remove_punct=False):
        """
        Builds a reverse mapping from strings to sets of entity IDs, handling duplicates.

        Args:
            entity_dict (dict): Original entity dictionary mapping entity IDs to sets/lists of strings.
            case_sensitive (bool): Whether the mapping should be case-sensitive.
            remove_punct (bool): Whether to remove punctuation from strings.

        Returns:
            dict: Reverse mapping from cleaned strings to sets of entity IDs.
        """
        reverse_mapping = defaultdict(set)
        for entity_id, names in entity_dict.items():
            for name in names:
                cleaned_name = name.strip()
                if remove_punct:
                    cleaned_name = self.remove_punctuation(cleaned_name)
                if not case_sensitive:
                    cleaned_name = cleaned_name.lower()
                reverse_mapping[cleaned_name].add(entity_id)

        logger.info(f"Reverse mapping created with {len(reverse_mapping)} unique strings.")
        return dict(reverse_mapping)

    def compute_or_load_relatedness_matrix(self, graph):
        if os.path.exists(self.relatedness_matrix_file):
            print("Loading relatedness matrix...")
            data = self.load_from_pickle(self.relatedness_matrix_file)
            return data['matrix'], data['predicates'], data['pred_to_idx']
        else:
            print("Computing relatedness matrix...")
            relatedness_matrix, predicates, predicates_to_idx = compute(graph, num_splits=1)
            self.save_to_pickle(
                {'matrix': relatedness_matrix, 'predicates': predicates, 'pred_to_idx': predicates_to_idx},
                self.relatedness_matrix_file
            )
            return relatedness_matrix, predicates, predicates_to_idx

    def preprocess_graph(self, graph):
        graph = graph[1:]  # Remove header row
        graph_df = pd.DataFrame(graph, columns=['s', 'p', 'o'])
        adjacency_list = defaultdict(list)
        for _, row in graph_df.iterrows():
            subject_id, relation_id, object_id = row['s'], row['p'], row['o']
            adjacency_list[subject_id].append((object_id, relation_id))
            adjacency_list[object_id].append((subject_id, relation_id))
        return adjacency_list

    def read_graph_from_gz(self, file_path):
        graph = []
        graph.append("s\tp\to")
        with os.open(file_path, 'rt') as file:
            next(file)
            for line in file:
                line = line.strip()
                subject_id, object_id, relation_id = line.split()
                graph.append((subject_id, object_id, relation_id))
        return graph

    def construct_dictionaries(self, kg_triples, entity_dict_name, relation_dict_name):


        return entity_dict, relation_dict

    def save_to_pickle(self, data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    def load_from_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def extract_focused_subgraph(self, query):
        query_subject, query_relation = query
        subgraph_nodes = set()
        subgraph_edges = []
        visited = set()
        queue = deque([(query_subject, 0)])

        while queue:
            current_node, depth = queue.popleft()
            if depth >= self.max_depth:
                break
            subgraph_nodes.add(current_node)
            neighbors = []
            for neighbor, relation in self.adjacency_list.get(current_node, []):
                similarity = self.relation_similarity_matrix[query_relation][self.pred_to_idx[relation]]
                if neighbor not in visited:
                    neighbors.append((neighbor, relation, similarity))
            k = max(self.min_k, int(len(neighbors) * self.k_fraction))
            top_neighbors = sorted(neighbors, key=lambda x: x[2], reverse=True)[:k]
            for neighbor, relation, similarity in top_neighbors:
                subgraph_edges.append((current_node, relation, neighbor))
                queue.append((neighbor, depth + 1))
                visited.add(neighbor)
        return subgraph_nodes, subgraph_edges

    def return_edges_nicely(self, edges):
        subgraph = []
        for subject_id, relation_id, object_id in edges:
            subject_name = self.entity_dict.get(subject_id, [f"Unknown({subject_id})"])[0]
            relation_name = self.relation_dict.get(relation_id, [f"Unknown({relation_id})"])[0]
            object_name = self.entity_dict.get(object_id, [f"Unknown({object_id})"])[0]
            subgraph.append((subject_name, relation_name, object_name))
        return subgraph

    def return_edges_subgraph(self, edges):
        """
        Converts edge IDs to human-readable names.

        Args:
            edges (list of tuples): List of edges in the format (subject_id, relation_id, object_id).

        Returns:
            list of tuples: List of edges in the format (subject_name, relation_name, object_name).
        """
        subgraph = []
        for subject_id, relation_id, object_id in edges:
            # Get the first alias/name for the entity; fallback to 'Unknown(ID)' if not found
            subject_names = self.entity_dict.get(subject_id, [f"Unknown({subject_id})"])
            subject_name = subject_names[0] if subject_names else f"Unknown({subject_id})"

            relation_names = self.relation_dict.get(relation_id, [f"Unknown({relation_id})"])
            relation_name = relation_names[0] if relation_names else f"Unknown({relation_id})"

            object_names = self.entity_dict.get(object_id, [f"Unknown({object_id})"])
            object_name = object_names[0] if object_names else f"Unknown({object_id})"

            # Append as a tuple
            subgraph.append((subject_name, relation_name, object_name))
        return subgraph


if __name__ == "__main__":
    # Example usage:
    dataset_name = "ZEROShotRE"
    max_depth = 1
    min_k = 1
    k_fraction = 0.6
    extractor = ExtractRelevantSubGraph(dataset_name, max_depth, min_k, k_fraction)

    # Sample query
    # Once everything is loaded, we can run a query:
    # For example, let's pick a predicate from extractor.pred_to_idx and query_id from entity_dict keys.
    # Note: This is just an example. Replace 'Q392' and 'P175' with known IDs in your dataset.
    query_id = 'Q392'
    predicate = 'P175'
    predicate_id = extractor.pred_to_idx[predicate]
    query = (query_id, predicate_id)

    nodes, edges = extractor.extract_focused_subgraph(query)
    nice_subgraph = extractor.return_edges_nicely(edges)
    print("Extracted Subgraph:", nice_subgraph)