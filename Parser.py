import json

##This allows, given a KB file and a question file, to create a json file with Question, subgraph, and answer.



def parse_kb_file_to_triples(kb_file_path):
    """
    Parse the KB file containing triples into the required format.
    Args:
        kb_file_path (str): Path to the KB file.
    Returns:
        list[tuple]: List of fact triples (subject, relation, object).
    """
    kb_triples = []
    with open(kb_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split the line into subject, relation, and object
            subject, relation, obj = line.split("\t")
            kb_triples.append((subject.strip(), relation.strip(), obj.strip()))
    return kb_triples


def integrate_data_with_kb_formatted(dataset, kb_triples):
    """
    Verify dataset subgraphs with KB triples to ensure validity.
    Args:
        dataset (list[dict]): Parsed dataset entries.
        kb_triples (list[tuple]): Parsed KB triples.
    Returns:
        bool: True if all paths are valid according to the KB.
    """
    kb_set = set(kb_triples)
    for entry in dataset:
        for triple in entry["subgraph"]:
            if tuple(triple) not in kb_set:
                print(f"Invalid triple in dataset: {triple}")
                return False
    return True

#
def parse_data_file_to_format(data_file_path, is_pql_format):
    """
    Parse the DATA file and format entries into the required structure.
    Args:
        data_file_path (str): Path to the DATA file.
        is_pql_format (bool): True if the dataset follows the PQL format.
    Returns:
        list[dict]: List of dataset entries in the desired format.
    """
    formatted_dataset = []
    with open(data_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Split the line into question, answer(s), and path
            parts = line.split("\t")
            if len(parts) < 3:
                print(f"Skipping invalid line: {line}")
                continue

            question = parts[0].strip()
            answer = parts[1].split("/")[0].strip()  # Use the first answer if multiple

            # Clean the answer by removing redundant parts like parentheses or duplicate words
            answer = answer.split("(")[0].strip()  # Remove anything after '('
            answer = answer.split(")")[0].strip()  # Remove anything after ')'
            answer = answer.replace("  ", " ").strip()  # Remove double spaces

            path = parts[2] if len(parts) > 2 else ""

            # Parse the path (subgraph)
            path_elements = path.split("#")
            subgraph = []
            for i in range(0, len(path_elements) - 1, 2):  # Process relation and entity pairs
                if path_elements[i] == "<end>" or path_elements[i + 1] == "<end>":
                    break  # Stop processing at <end>
                if i + 2 < len(path_elements):
                    subject = path_elements[i].strip()
                    relation = path_elements[i + 1].strip()
                    obj = path_elements[i + 2].strip()

                    # Ensure no <end> or other invalid tokens in triples
                    if subject == "<end>" or relation == "<end>" or obj == "<end>":
                        continue

                    # Append valid triples only
                    subgraph.append([subject, relation, obj])

            # Append the formatted entry
            if subgraph:
                formatted_dataset.append({
                    "question": question,
                    "subgraph": subgraph,
                    "answer": answer
                })
            else:
                print(f"Invalid or empty subgraph in line: {line}")

    return formatted_dataset


# Example usage
base_path= "datasets/PathQuestion/"
dataset="PQ-3H"
# Example usage
data_file_path = base_path+dataset+".txt"  # Replace with your data file path
kb_file_path = "/3H-KB.txt"  # Replace with your KB file path

is_pql_format = True  # Set to False for PQ format

# Parse the files
formatted_dataset = parse_data_file_to_format(data_file_path, is_pql_format)
kb_triples = parse_kb_file_to_triples(kb_file_path)

# Validate and print the formatted dataset
if integrate_data_with_kb_formatted(formatted_dataset, kb_triples):
    print("All dataset paths are valid according to the KB.")
else:
    print("Some dataset paths are invalid.")

# Save the formatted dataset to a JSON file
output_file_path = base_path+dataset+"/"+dataset+".json"
print(output_file_path)

with open(output_file_path, "w") as f:
    json.dump(formatted_dataset, f, indent=4)
print(f"Formatted dataset saved to {output_file_path}.")
