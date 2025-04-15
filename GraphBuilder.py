import os
import pickle
import string
from collections import defaultdict
import pandas as pd
import logging

from PredicateRelatednessMatrix import compute

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        # Set base paths
        self.base_path = "/"  # Adjust if needed
        self.data_path = os.path.join(self.base_path) + "/"

        # File paths
        self.graph_pickle = os.path.join(self.data_path, 'graph.pkl')
        self.entity_dict_pickle = os.path.join(self.data_path, 'entity_dict.pkl')
        self.entity_dict_reverse_pickle = os.path.join(self.data_path, 'entity_reverse_dict.pkl')
        self.relation_dict_pickle = os.path.join(self.data_path, 'relation_dict.pkl')
        self.relation_reverse_dict_pickle = os.path.join(self.data_path, 'relation_reverse_dict.pkl')
        self.relatedness_matrix_file = os.path.join(self.data_path, 'relatedness_matrix.pkl')

        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)

    def build_graph(self, triple_file_path):
        if os.path.exists(self.graph_pickle):
            logger.info("Loading processed graph data...")
            graph, adjacency_list = self.load_from_pickle(self.graph_pickle)
            entity_dict = self.load_from_pickle(self.entity_dict_pickle)
            entity_reverse_dict = self.load_from_pickle(self.entity_dict_reverse_pickle)
            relation_dict = self.load_from_pickle(self.relation_dict_pickle)
            relation_reverse_dict = self.load_from_pickle(self.relation_reverse_dict_pickle)
        else:
            logger.info("Reading raw data and constructing graph...")
            graph = self.read_triples(triple_file_path)
            adjacency_list = self.preprocess_graph(graph)
            self.save_to_pickle([graph, adjacency_list], self.graph_pickle)
            entity_dict, relation_dict = self.build_dictionaries(graph)
            entity_reverse_dict = self.build_reverse_mapping_multiple(entity_dict)
            relation_reverse_dict = self.build_reverse_mapping_multiple(relation_dict)

            self.save_to_pickle(entity_dict, self.entity_dict_pickle)
            self.save_to_pickle(entity_reverse_dict, self.entity_dict_reverse_pickle)
            self.save_to_pickle(relation_dict, self.relation_dict_pickle)
            self.save_to_pickle(relation_reverse_dict, self.relation_reverse_dict_pickle)


        return graph, adjacency_list, entity_dict, entity_reverse_dict, relation_dict,relation_reverse_dict

    def compute_relatedness_matrix(self, graph):
        if os.path.exists(self.relatedness_matrix_file):
            logger.info("Loading relatedness matrix...")
            data = self.load_from_pickle(self.relatedness_matrix_file)
            return data['matrix'], data['predicates'], data['pred_to_idx']
        else:
            logger.info("Computing relatedness matrix...")
            relatedness_matrix, predicates, predicates_to_idx = compute(graph, self.relatedness_matrix_file)
            self.save_to_pickle(
                {'matrix': relatedness_matrix, 'predicates': predicates, 'pred_to_idx': predicates_to_idx},
                self.relatedness_matrix_file
            )
            return relatedness_matrix, predicates, predicates_to_idx

    @staticmethod
    def remove_punctuation(text):
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def build_reverse_mapping_multiple(self, entity_dict, case_sensitive=True, remove_punct=False):
        reverse_mapping = defaultdict(set)
        for entity_id, name in entity_dict.items():
            reverse_mapping[name].add(entity_id)

        logger.info(f"Reverse mapping created with {len(reverse_mapping)} unique strings.")
        return dict(reverse_mapping)

    def preprocess_graph(self, graph):
        if isinstance(graph, list) and len(graph) > 0 and isinstance(graph[0], str):
            graph = graph[1:]  # Remove header if present
        graph_df = pd.DataFrame(graph, columns=['s', 'p', 'o'])
        adjacency_list = defaultdict(list)
        for _, row in graph_df.iterrows():
            subject_id, relation_id, object_id = row['s'], row['p'], row['o']
            adjacency_list[subject_id].append((object_id, relation_id))
            adjacency_list[object_id].append((subject_id, relation_id))
        return adjacency_list

    def read_triples(self, file_path):
        triples = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                parts = line.split("|") #It may vary depending on the dataset
                if len(parts) != 3:
                    print(f"Warning: Line {line_number} is malformed: '{line}'")
                    continue  # Skip malformed lines
                subject, predicate, object_ = [part.strip() for part in parts]
                triples.append((subject, predicate, object_))
        return triples

    def build_dictionaries(self, triples):
        # Extract unique entities and relations
        entities = set()
        relations = set()
        for subj, pred, obj in triples:
            entities.add(subj)
            entities.add(obj)
            relations.add(pred)

        # Assign unique IDs to entities starting from 1
        entity_dict = {entity: idx for idx, entity in enumerate(sorted(entities), start=0)}

        # Assign unique IDs to relations starting from 0
        relation_dict = {relation: idx for idx, relation in enumerate(sorted(relations), start=0)}

        return entity_dict,relation_dict

    @staticmethod
    def save_to_pickle(data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {file_path}")

    @staticmethod
    def load_from_pickle(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded from {file_path}")
        return data

if __name__ == "__main__":
    # Example usage:
    dataset_name = "dataset/PathQuestion/"
    triple_file_path = "PQ-2H-KB.txt"  # Replace with actual path

    builder = GraphBuilder(dataset_name)
    graph, adjacency_list, entity_dict, entity_reverse_dict, relation_dict, relation_reverse_dict = builder.build_graph(
        triple_file_path
    )

    relatedness_matrix, predicates, pred_to_idx = builder.compute_relatedness_matrix(graph)

    logger.info("Graph construction and relatedness matrix computation completed.")