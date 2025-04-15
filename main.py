import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data.remote_backend_utils import num_nodes
from torch_geometric.loader import DataLoader
from torch_geometric.nn import RGCNConv, global_mean_pool
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig,AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import seaborn as sns

from relational_graph_token.datasets.QuestionDataset import *

# device = torch.device("mps" if torch.has_mps else "cpu")
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(f"Using device: {device}")

#
# Simple GCN
#

class GraphEncoder(nn.Module):
    def __init__(self, n_nodes, n_rel, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.entity_embedding = nn.Embedding(n_nodes, input_dim)
        self.relation_embedding = nn.Embedding(n_rel, input_dim)
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=n_rel)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations=n_rel)

    def forward(self, h, r, t):
        # Get node features for all possible nodes
        x = self.entity_embedding.weight  # [n_nodes, input_dim]

        # Use h and t directly as edge indices
        edge_index = torch.stack([h, t], dim=0)  # [2, E]

        # Run RGCN layers
        x = self.conv1(x, edge_index, r).relu()
        x = self.conv2(x, edge_index, r)  # [n_nodes, output_dim]

        # Aggregate node embeddings into a single graph embedding
        graph_embedding = x.mean(dim=0, keepdim=True)  # [1, output_dim]

        return graph_embedding

#
# Relational GCN
#
class EnhancedGraphEncoder(nn.Module):
    def __init__(self, n_nodes, n_rel, hidden_dim, output_dim, llm_dim=768):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_rel = n_rel
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.llm_dim = llm_dim  # Dimension required to align with LLM

        # Entity and relation embeddings (will be initialized dynamically)
        self.entity_embedding = None
        self.relation_embedding = None

        # Initial projection layer to align node features with LLM-compatible dimension
        self.initial_proj = nn.Linear(hidden_dim, llm_dim)

        # Projection layer for graph embeddings
        self.graph_proj = nn.Linear(hidden_dim, llm_dim)

        # RGCN layers with residual connections
        self.conv1 = RGCNConv(llm_dim, hidden_dim, num_relations=n_rel)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=n_rel)

        # Relation-aware attention
        self.relation_attention = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.Tanh(),
            nn.Linear(llm_dim, 1)
        )

        # Hierarchical pooling components
        self.local_pool = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.ReLU(),
            nn.Linear(llm_dim, 1)
        )

        # Final projection to match output dimension
        self.output_proj = nn.Sequential(
            nn.Linear(llm_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def initialize_embeddings(self, input_dim, device):
        """
        Dynamically initialize embeddings based on input_dim.
        """
        # print(f"Initializing embeddings with input_dim={input_dim}")
        # Initialize entity and relation embeddings
        self.entity_embedding = nn.Embedding(self.n_nodes, input_dim).to(device)
        self.relation_embedding = nn.Embedding(self.n_rel, input_dim).to(device)

    def forward(self, x, edge_index, edge_attr, batch, ptr):
        """
        Forward pass for batched graphs with per-graph relation context.
        Args:
            x: Node features [total_nodes_in_batch, input_dim]
            edge_index: Edge indices [2, total_edges_in_batch]
            edge_attr: Edge attributes [total_edges_in_batch] (relation indices)
            batch: Batch tensor mapping nodes to graphs [total_nodes_in_batch]
            ptr: Offset tensor defining graph boundaries
        Returns:
            Tuple of:
            - Node embeddings [total_nodes_in_batch, hidden_dim]
            - Graph representations [batch_size, output_dim]
        """
        device = x.device

        # Dynamically initialize embeddings if not yet initialized
        if self.entity_embedding is None:
            input_dim = x.size(-1)
            self.initialize_embeddings(input_dim, device)

        # Project node features
        x = self.initial_proj(x)

        # Apply RGCN layers
        x1 = self.dropout(F.relu(self.conv1(x, edge_index, edge_attr)))
        x = x1 + self.dropout(F.relu(self.conv2(x1, edge_index, edge_attr)))

        # Store node embeddings
        node_embeddings = x

        # Correct batch assignment
        batch_correct = torch.repeat_interleave(
            torch.arange(len(ptr) - 1, device=device),
            torch.diff(ptr).to(device)
        )

        # Pool nodes to get graph embeddings
        graph_embeddings = global_mean_pool(x, batch_correct)  # [batch_size, hidden_dim]
        graph_embeddings = self.graph_proj(graph_embeddings)  # [batch_size, llm_dim]

        # Get edge batch assignment using source nodes
        edge_batch = batch_correct[edge_index[0]]  # [total_edges_in_batch]
        num_graphs = len(ptr) - 1

        # Compute relation embeddings
        rel_embeds = self.relation_embedding(edge_attr).to(device)  # [total_edges_in_batch, input_dim]
        rel_embeds = self.initial_proj(rel_embeds)  # [total_edges_in_batch, llm_dim]
        # Compute attention scores
        rel_attention = F.softmax(self.relation_attention(rel_embeds), dim=0)  # [total_edges_in_batch, 1]
        # Initialize tensor for per-graph relation contexts
        rel_contexts = torch.zeros(num_graphs, rel_embeds.size(-1), device=device)  # [num_graphs, llm_dim]

        # Compute relation context for each graph separately
        for graph_idx in range(num_graphs):
            # Get mask for edges in this graph
            graph_mask = (edge_batch == graph_idx)

            if graph_mask.any():
                # Get relation embeddings and attention scores for this graph
                graph_rel_embeds = rel_embeds[graph_mask]  # [num_edges_in_graph, llm_dim]
                graph_rel_attention = rel_attention[graph_mask]  # [num_edges_in_graph, 1]
                # Normalize attention scores for this graph
                graph_rel_attention = F.softmax(graph_rel_attention, dim=0)
                # Compute weighted sum for this graph
                rel_contexts[graph_idx] = torch.sum(
                    graph_rel_embeds * graph_rel_attention, dim=0
                )  # [llm_dim]

        # Combine global and local representations
        combined_repr = torch.cat([
            graph_embeddings,  # Global graph context
            rel_contexts  # Per-graph relation context
        ], dim=-1)  # [batch_size, llm_dim * 2]

        # Project to final dimension
        combined_repr = self.output_proj(combined_repr)  # [batch_size, output_dim]

        return node_embeddings, combined_repr

class EnhancedKGTokenLLM(nn.Module):
    def __init__(self, num_entities, num_relations, hidden_dim=980, llm_name="gpt2", top_k_attention_nodes=4,
                 device="mps"):
        super().__init__()
        # Set a custom timeout for Hugging Face downloads
        import os
        os.environ["HUGGINGFACE_HUB_DEFAULT_TIMEOUT"] = "300"  # 5-minute timeout


        # Determine if the model is T5-based (e.g., FLAN) or Causal LM
        is_flan = "flan" in llm_name.lower()
        # Obtain the token dimension (embedding size) for the specific LLM
        self.token_dim = self.get_token_dim(llm_name)
        # Load the model configuration
        config = AutoConfig.from_pretrained(llm_name)
        max_positions = getattr(config, "n_positions", None) or getattr(config, "max_position_embeddings", None)
        print(
            f"Token dimension: {self.token_dim}. Max tokens the model can handle: {max_positions or 'unknown (check model docs)'}")
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name, local_files_only=True)

        # Set pad token explicitly if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load and freeze LLM
        if is_flan:
            print(f"Loading FLAN (Seq2Seq) model: {llm_name}")
            self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name, local_files_only=True)#.to(device)
        else:
            print(f"Loading Causal Language Model: {llm_name}")
            self.llm = AutoModelForCausalLM.from_pretrained(llm_name, local_files_only=True).to(device)

        # Freeze LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Number of nodes per graph (can be set dynamically later)
        self.num_nodes_per_graph = None

        # Initialize GraphEncoder
        self.graph_encoder = EnhancedGraphEncoder(num_entities, num_relations, hidden_dim, self.token_dim).to(device)

        # Get frozen LLM embeddings (used for graph-token alignment)
        self.llm_embeddings = self.llm.get_input_embeddings()  # Frozen

        # Number of top attention nodes to select (for attention-based filtering)
        self.top_attention_nodes = top_k_attention_nodes

    def get_token_dim(self, model_name):
        """
        Retrieve the token embedding size (hidden dimension) for the specified model.

        Args:
            model_name (str): The name of the model.

        Returns:
            int: The token embedding size (hidden dimension).
        """
        config = AutoConfig.from_pretrained(model_name)

        # Handle different architectures dynamically
        if hasattr(config, "hidden_size"):  # For models like GPT, BERT, etc.
            return config.hidden_size
        elif hasattr(config, "d_model"):  # For models like T5 (FLAN)
            return config.d_model
        else:
            raise ValueError(f"Token dimension not found for model: {model_name}")

    def attention_based_readout(self, node_embeddings, question_embeddings, top_attention_nodes):
        """
        Use attention mechanism to select the most relevant nodes for each question in the batch.
        Args:
            node_embeddings: Node embeddings [batch_size, num_nodes, llm_dim].
            question_embeddings: Question embeddings [batch_size, 1, llm_dim].
            top_attention_nodes: Number of top nodes to select.
        Returns:
            Selected node embeddings [batch_size, top_k, llm_dim].
        """
        # Ensure compatibility of dimensions
        if node_embeddings.size(-1) != question_embeddings.size(-1):
            raise ValueError(
                f"Node embeddings dimension ({node_embeddings.size(-1)}) does not match "
                f"question embeddings dimension ({question_embeddings.size(-1)})!"
            )

        # Compute attention scores
        attention_scores = torch.matmul(node_embeddings, question_embeddings.transpose(1, 2))  # [batch_size, num_nodes, 1]

        valid_mask = torch.arange(attention_scores.size(1), device=device).unsqueeze(0) < self.num_nodes_per_graph.unsqueeze(
            1)
        masked_attention_scores = attention_scores.squeeze(-1).masked_fill(~valid_mask, float('-inf'))
        top_k_indices = torch.topk(masked_attention_scores, top_attention_nodes, dim=-1).indices


        # Gather embeddings of top-k nodes for each question in the batch
        selected_embeddings = torch.stack([
            node_embeddings[i, top_k_indices[i]]
            for i in range(node_embeddings.size(0))
        ])  # [batch_size, top_k, llm_dim]

        return selected_embeddings, top_k_indices

    def forward(self, batch, answer_id=None):
        """
        Forward pass combining graph tokens, questions, and optional answers.
        Args:
            batch: Batched data object from PyTorch Geometric DataLoader.
            answer_id: Optional tensor of answers for supervised learning.
        """
        if isinstance(batch, tuple):  # Handle tuple batch
            batch = batch[0]

        if not hasattr(batch, "num_graphs"):
            raise ValueError("Expected batch to be a PyTorch Geometric Batch object with a `num_graphs` attribute.")

        batch_size = batch.num_graphs  # Number of graphs in the batch
        node_features = batch.x  # [total_nodes_in_batch, feature_dim]
        edge_index = batch.edge_index  # [2, total_edges_in_batch]
        edge_attr = batch.edge_attr  # [total_edges_in_batch] (relation types)
        ptr = batch.ptr  # Offset tensor for graph boundaries

        self.num_nodes_per_graph=torch.diff(ptr)

        if edge_attr.dim() > 1:
            edge_attr = edge_attr.argmax(dim=1)  # Convert to relation indices

        # Encode graph into node and graph embeddings
        node_embeddings, global_graph_embedding = self.graph_encoder(
            node_features, edge_index, edge_attr, batch.batch, ptr
        )  # node_embeddings: [total_nodes, token_dim], global_graph_embedding: [batch_size, token_dim]


        # Step 2: Attention-based readout
        # Ensure consistent embedding dimensions
        node_embeddings_split = node_embeddings.split(torch.diff(ptr).tolist(), dim=0)  # Split by graph
        max_nodes = max(len(nodes) for nodes in node_embeddings_split)

        # Create padded node embeddings
        padded_node_embeddings = torch.zeros((batch_size, max_nodes, self.token_dim), device=device)
        for i, nodes in enumerate(node_embeddings_split):
            # Project nodes to match `self.token_dim` if necessary
            if nodes.size(1) != self.token_dim:
                nodes = F.pad(nodes, (0, self.token_dim - nodes.size(1)))
            padded_node_embeddings[i, :len(nodes)] = nodes

        # Tokenize questions
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.llm.resize_token_embeddings(len(self.tokenizer))
        # Tokenize questions
        question_ids = self.tokenizer(batch.question, padding=True, return_tensors="pt")["input_ids"].to(device)
        question_embeddings = self.llm_embeddings(question_ids).mean(dim=1, keepdim=True)  # [batch_size, 1, token_dim]

        # Compute attention scores and select top-k nodes
        selected_node_embeddings, _ = self.attention_based_readout(
            padded_node_embeddings, question_embeddings, top_attention_nodes=self.top_attention_nodes
        )

        # Tokenize delimiters
        graph_delimiter_ids = self.tokenizer("[GRAPH]", add_special_tokens=False, return_tensors="pt").input_ids.to(
            device)
        node_delimiter_ids = self.tokenizer("[NODES]", add_special_tokens=False, return_tensors="pt").input_ids.to(
            device)
        question_delimiter_ids = self.tokenizer("[QUESTION]", add_special_tokens=False,
                                                return_tensors="pt").input_ids.to(device)
        answer_delimiter_ids = self.tokenizer("[ANSWER]", add_special_tokens=False, return_tensors="pt").input_ids.to(
            device)

        # Compute embeddings for delimiters
        graph_delimiter_embeds = self.llm_embeddings(graph_delimiter_ids).mean(dim=1, keepdim=True)
        node_delimiter_embeds = self.llm_embeddings(node_delimiter_ids).mean(dim=1, keepdim=True)
        question_delimiter_embeds = self.llm_embeddings(question_delimiter_ids).mean(dim=1, keepdim=True)
        answer_delimiter_embeds = self.llm_embeddings(answer_delimiter_ids).mean(dim=1, keepdim=True)

        # Directly use answer IDs as labels
        if answer_id is not None:
            labels = answer_id.to(device)  # Ensure the IDs are on the correct device
            # Ensure labels are 1D with batch size alignment
            if labels.dim() == 2:
                labels = labels.squeeze(1)
        else:
            labels = None

        # Concatenate embeddings
        combined_embeds = torch.cat([
            graph_delimiter_embeds.expand(batch_size, -1, -1),
            global_graph_embedding.unsqueeze(1).expand(batch_size, 1, -1),
            node_delimiter_embeds.expand(batch_size, -1, -1),
            selected_node_embeddings,  # [batch_size, top_k, token_dim]
            question_delimiter_embeds.expand(batch_size, -1, -1),
            question_embeddings,  # [batch_size, 1, token_dim]
            answer_delimiter_embeds.expand(batch_size, -1, -1),
        ], dim=1)  # [batch_size, seq_len, token_dim]

        seq_len = combined_embeds.size(1)  # Get the sequence length of the combined embeddings

        # Update labels
        if answer_id is not None:
            # Create a mask for non-answer tokens
            non_answer_mask = torch.full((batch_size, seq_len - 1), -100, dtype=torch.long, device=device)
            # Append answer_id as the last token
            labels = torch.cat([non_answer_mask, answer_id.unsqueeze(1)], dim=1)  # [batch_size, seq_len]
        else:
            labels = None

        # Verify alignment
        assert labels is None or labels.size(0) == combined_embeds.size(0), (
            f"Mismatch: labels batch size ({labels.size(0)}) must match combined_embeds batch size ({combined_embeds.size(0)})."
        )
        assert labels is None or labels.size(1) == combined_embeds.size(1), (
            f"Mismatch: labels sequence length ({labels.size(1)}) must match combined_embeds sequence length ({combined_embeds.size(1)})."
        )


        # Forward through the LLM
        outputs = self.llm(
            inputs_embeds=combined_embeds,
            attention_mask=torch.ones(combined_embeds.size()[:-1], dtype=torch.long, device=device),
            labels=labels if labels is not None else None,
            output_hidden_states=True
        )
        return outputs

    def generate_answer(self, outputs, graph_entities):
        """
        Generate text answer from LLM logits constrained to graph entities.
        """
        logits = outputs.logits[:, -1, :]
        entity_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(e) for e in graph_entities]).to(logits.device)
        entity_logits = logits[:, entity_ids]
        predicted_entity_id = entity_logits.argmax(dim=-1)
        return self.tokenizer.decode(entity_ids[predicted_entity_id])

def compute_loss_and_perplexity(outputs):
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return loss, perplexity



def load_dataset_and_model(base_path, dataset_name, llm, seed, train_ratio=0.8, val_ratio=0.1):
    root_dir = base_path + dataset_name + "/padded_graph_dataset"
    dataset = QuestionDataset(root=root_dir)

    # Infer input_dim (hidden_dim) from the dataset
    sample_data = dataset[0]
    hidden_dim = sample_data.x.size(1)

    # Calculate the number of samples for each split
    num_total = len(dataset)
    num_train = int(train_ratio * num_total)
    num_val = int(val_ratio * num_total)
    num_test = num_total - num_train - num_val

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Testing set size: {len(test_dataset)}")

    # Load mappings
    mappings_path = f"{root_dir}/processed/mappings.pt"
    mappings = torch.load(mappings_path)
    entity_to_id = mappings["entity_to_id"]
    relation_to_id = mappings["relation_to_id"]

    n_entities = len(entity_to_id)
    n_relations = len(relation_to_id)

    # Initialize the model
    model = EnhancedKGTokenLLM(
        num_entities=n_entities,
        hidden_dim=hidden_dim,
        num_relations=n_relations,
        llm_name=llm
    ).to(device)

    return train_dataset, val_dataset, test_dataset, model, entity_to_id, relation_to_id

def train_model(model, train_loader, val_loader, checkpoint_path, epochs=1000, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_val_loss = float('inf')
    patience_counter = 0

    print("Start training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, answer_id=batch.answer_id)
            loss, perplexity = compute_loss_and_perplexity(outputs)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch, answer_id=batch.answer_id)
                loss, perplexity = compute_loss_and_perplexity(outputs)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved at {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

def evaluate_model(model, test_loader, entity_to_id, relation_to_id, top_k=10):
    print("Start testing...")
    model.eval()

    total_samples = 0
    correct_at_k = 0
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}
    valid_ids = set(entity_to_id.values()).union(set(relation_to_id.values()))
    all_decoded_tokens=[]
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, tuple):
                batch, answers = batch
            else:
                answers = batch.answer_id
            batch = batch.to(device)
            questions = batch.question
            outputs = model(batch)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            valid_ids_list = list((valid_ids))
            for i in range(probs.size(0)):
                graph_probs = probs[i]
                valid_probs = graph_probs[valid_ids_list]
                top_k_probs, top_k_indices = torch.topk(valid_probs, top_k)
                top_k_ids = [valid_ids_list[idx.item()] for idx in top_k_indices]
                ##
                correct_answer = answers[i].item()
                # Decode tokens using the tokenizer
                top_k_token_ids = top_k_indices.tolist()  # Assuming batch_size = 1

                if correct_answer in top_k_ids:
                    correct_at_k += 1
                print(f"Graph {i + 1}/{probs.size(0)}")
                print(f"  Question: {questions[i]}")
                print(f"  Correct Answer ID: {correct_answer}")
                print(f"  Top Predictions IDs: {top_k_ids[:top_k]}")
                print(f"  Correct Answer String: {id_to_entity.get(correct_answer)}")
            total_samples += probs.size(0)

    precision_at_k = correct_at_k / total_samples if total_samples > 0 else 0
    print(f"Precision@{top_k}: {precision_at_k:.4f}")

    unique_elements = list(set(element for sublist in all_decoded_tokens for element in sublist))

#
def main(base_path, dataset_name, llm, seed=42, epochs=1000, batch_size=16, top_k=10):

    train_dataset, val_dataset, test_dataset, model, entity_to_id, relation_to_id = load_dataset_and_model(
        base_path, dataset_name, llm, seed
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    checkpoint_path = f"{base_path}/{dataset_name}/best_model.pth"

    # Train the model
    train_model(model, train_loader, val_loader, checkpoint_path, epochs)

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))

    # Evaluate the model
    evaluate_model(model, test_loader, entity_to_id, relation_to_id, top_k)

if __name__ == "__main__":

    base_path ="datasets/" 

    dataset_name = "/PathQuestion/PQ-2H"  # Replace with your dataset name
    seed=79
    epochs = 100
    train_ratio = 0.8
    val_ratio = 0.1
    batch_size = 64
    precision_at_K = 5 #K value
    llm="gpt2" #gpt2 #bert-base-uncased #gpt2-medium #flan-t5-xl #flan-t5-xxl

    main(
        base_path=base_path,
        dataset_name=dataset_name,
        llm=llm,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        top_k=precision_at_K
    )
