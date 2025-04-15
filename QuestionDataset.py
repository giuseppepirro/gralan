import json

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data

#
# This will create Pytorch Geometric datasets for training in the format: Subgraph-Question-Answer. Will also pad graphs
#

# ------------------ LAPLACIAN POSITIONAL EMBEDDING ------------------

def laplacian_pos_embedding(graph: nx.Graph, units: int = 4) -> nx.Graph:
    """Adds Laplacian positional encoding to the graph."""
    m = nx.normalized_laplacian_matrix(
        graph, nodelist=sorted(graph.nodes), weight=None
    ).astype(np.float32)

    u, _, _ = np.linalg.svd(m.todense(), compute_uv=True)


    if units > u.shape[1]:
        u = np.pad(u, ((0, 0), (0, units - u.shape[1])))
    nx.set_node_attributes(
        graph, dict(zip(sorted(graph.nodes), u[:, :units])), name='lpe'
    )
    return graph

# ------------------ DATA PREPROCESSING ------------------

def preprocess_dataset_with_lpe(dataset):
    """
    Process dataset to include Laplacian positional embeddings (LPE).
    Args:
        dataset: List of graph data (e.g., JSON format with question, subgraph, and answer).
    Returns:
        processed_data: List of dictionaries with processed subgraphs and features.
        entity_to_id: Mapping of entity names to unique IDs.
        relation_to_id: Mapping of relation names to unique IDs.
    """
    entity_to_id = {}
    relation_to_id = {}
    processed_data = []

    for entry in dataset:
        nodes = set()
        edges = []
        nx_graph = nx.Graph()

        for s, r, t in entry["subgraph"]:
            # Map entities and relations to unique IDs
            if s not in entity_to_id:
                entity_to_id[s] = len(entity_to_id)
            if t not in entity_to_id:
                entity_to_id[t] = len(entity_to_id)
            if r not in relation_to_id:
                relation_to_id[r] = len(relation_to_id)

            # Add edges to graph
            edges.append((entity_to_id[s], relation_to_id[r], entity_to_id[t]))
            nx_graph.add_edge(entity_to_id[s], entity_to_id[t])

            # Add nodes
            nodes.add(entity_to_id[s])
            nodes.add(entity_to_id[t])

        # Compute LPE
        nx_graph = laplacian_pos_embedding(nx_graph)
        node_list = sorted(nx_graph.nodes())
        lpe = np.stack([nx_graph.nodes[node]["lpe"] for node in node_list], axis=0)

        # Determine if answer is an entity or relation
        answer = entry["answer"]
        if answer in entity_to_id:
            answer_id = entity_to_id[answer]
            answer_type = "entity"

            # Store processed data
            processed_data.append({
                "question": entry["question"],
                "subgraph": edges,
                "nodes": list(nodes),
                "lpe": lpe,  # Store LPE as part of the node features
                "answer": answer,  # Store mapped answer ID
                "answer_type": answer_type,  # Store whether the answer is an entity or relation
            })
    return processed_data, entity_to_id, relation_to_id


def create_pyg_data_with_lpe(processed_data, num_entities, num_relations):
    """
    Create PyTorch Geometric Data objects from processed data with LPE.
    Args:
        processed_data: List of processed subgraphs with LPE.
        num_entities: Total number of unique entities in the dataset.
        num_relations: Total number of unique relations in the dataset.
    Returns:
        List of PyTorch Geometric Data objects.
    """
    graph_data = []

    relation_embeddings = torch.eye(num_relations)
    size=len(processed_data)
    done=0
    for entry in processed_data:
        nodes = entry["nodes"]
        edges = entry["subgraph"]
        lpe = entry["lpe"]  # Laplacian positional embedding

        # Node features: Combine one-hot entity embedding with LPE
        entity_features = torch.eye(num_entities)[nodes]  # One-hot encoding
        lpe_features = torch.tensor(lpe, dtype=torch.float32)  # LPE features
        x = torch.cat([entity_features, lpe_features], dim=-1)  # Combine features

        # Edge index and attributes
        edge_index = torch.tensor([(h, t) for h, _, t in edges], dtype=torch.long).t()  # [2, num_edges]
        edge_attr = torch.tensor([r for _, r, _ in edges], dtype=torch.long)  # [num_edges], relation indices

        # Do not modify `edge_attr` for RGCNConv
        # If embeddings are required elsewhere, keep them separate
        edge_attr_emb = relation_embeddings[edge_attr]  # This should NOT replace edge_attr

        # Create PyG Data object and include the `question` attribute
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr_emb,
            answer=entry["answer"],  # Answer as label
            question=entry["question"],  # Include question
        )
        done=done+1
        if done%100==0:
            print("DONE ",done, " OUT OF ",size)

        graph_data.append(data)

    return graph_data

# ------------------ GRAPH PADDING ------------------

class GraphPadder:
    def __init__(self, slack: float = 1.0):
        self.slack = slack
        self.max_nodes = 0
        self.max_edges = 0

    def calculate_padding_statistics(self, dataset):
        node_counts = [data.num_nodes for data in dataset]
        edge_counts = [data.num_edges for data in dataset]

        self.max_nodes = int(max(node_counts) + self.slack * np.std(node_counts) + 1)
        self.max_edges = int(max(edge_counts) + self.slack * np.std(edge_counts) + 1)

    def pad_graph(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        padded_x = torch.zeros((self.max_nodes, data.x.size(1)), device=data.x.device)
        padded_x[:num_nodes] = data.x

        num_edges = data.edge_index.size(1)
        padded_edge_index = torch.full(
            (2, self.max_edges), self.max_nodes - 1, device=data.edge_index.device, dtype=torch.long
        )
        padded_edge_index[:, :num_edges] = data.edge_index

        if data.edge_attr is not None:
            padded_edge_attr = torch.zeros((self.max_edges, data.edge_attr.size(1)), device=data.edge_attr.device)
            padded_edge_attr[:num_edges] = data.edge_attr
        else:
            padded_edge_attr = None

        node_mask = torch.zeros(self.max_nodes, dtype=torch.bool, device=data.x.device)
        node_mask[:num_nodes] = 1

        edge_mask = torch.zeros(self.max_edges, dtype=torch.bool, device=data.edge_index.device)
        edge_mask[:num_edges] = 1

        return Data(
            x=padded_x,
            edge_index=padded_edge_index,
            edge_attr=padded_edge_attr,
            y=data.y,
            node_mask=node_mask,
            edge_mask=edge_mask
        )

    def pad_batch(self, dataset):
        return [self.pad_graph(data) for data in dataset]

# ------------------ PyG DATASET ------------------

from torch_geometric.data import Data

class QuestionDataset(InMemoryDataset):
    def __init__(self, root, dataset_json=None, transform=None, pre_transform=None, slack=1.0):
        self.dataset_json = dataset_json  # Pass dataset JSON for initial processing
        self.slack = slack
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def processed_file_names(self):
        # Return the name of the processed file(s)
        return ["padded_graph_data.pt"]

    def process(self):
        if self.dataset_json is None:
            raise ValueError("No dataset JSON provided for processing!")

        # Step 1: Preprocess dataset
        processed_data, entity_to_id, relation_to_id = preprocess_dataset_with_lpe(self.dataset_json)

        # Step 2: Create PyG Data
        graph_data = create_pyg_data_with_lpe(processed_data, len(entity_to_id), len(relation_to_id))

        # Step 3: Pad Graphs
        padder = GraphPadder(slack=self.slack)
        padder.calculate_padding_statistics(graph_data)
        padded_graph_data = padder.pad_batch(graph_data)

        # Add 'question' to each Data object
        for data, entry in zip(padded_graph_data, processed_data):
            data.question = entry["question"]  # Save the question as a custom attribute

        # Add 'answer' to each Data object
        for data, entry in zip(padded_graph_data, processed_data):
            data.answer = entry["answer"]  # Save the question as a custom attribute

        # Add 'answer.ids' to each Data object
        for data, entry in zip(padded_graph_data, processed_data):
            data.answer_id = entity_to_id.get(entry["answer"])  # Save the question as a custom attribute
            #data.answer_id = entity_to_id.get(entry["answer"], -1)  # Use a default ID if not found

        # Save processed data
        data, slices = self.collate(padded_graph_data)
        torch.save((data, slices), self.processed_paths[0])

        # Save additional mappings
        mappings = {
            "entity_to_id": entity_to_id,
            "relation_to_id": relation_to_id,
        }
        torch.save(mappings, self.processed_paths[0].replace("padded_graph_data.pt", "mappings.pt"))


# ------------------ EXAMPLE USAGE ------------------

# Load JSON dataset
def load_json_dataset(file_path):
    with open(file_path, "r") as f:
        dataset = json.load(f)
    # Convert subgraph tuples back from lists if needed
    for entry in dataset:
        entry["subgraph"] = [tuple(triple) for triple in entry["subgraph"]]

    return dataset#[0:10000]


def main():
    # Define paths and dataset information
 
    base_path = "datasets/"

    dataset_name = "PathQuestion"
    subfile="PQ-3H"
    file_path = base_path + dataset_name + "/"+subfile+"/"+subfile+".json"

    # Load the dataset
    print("Loading dataset...")
    dataset_json = load_json_dataset(file_path)

    # Create the PyG Dataset
    root_dir = base_path + dataset_name + "/"+subfile+ "/padded_graph_dataset"
    print("Creating PyG dataset...")
    question_dataset = QuestionDataset(root=root_dir, dataset_json=dataset_json)

    # Access a sample graph
    print(f"Number of graphs: {len(question_dataset)}")
    data = question_dataset[0]
    print("Sample Graph Information:")
    print(f"  Question: {data.question}")
    print(f"  Answer: {data.answer}")
    print(f"  Answer ID: {data.answer_id}")
    print(f"  Graph Nodes: {data.x.shape}")
    print(f"  Graph Edges: {data.edge_index.shape}")


if __name__ == "__main__":
    main()
