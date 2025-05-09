#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs K-means clustering on IDF scores and adds cluster labels to the original data.
Also assigns a unique integer ID to each unique dataset name as a task ID.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import argparse
import os


def load_data(input_file):
    """
    Load data from a JSONL file and extract IDF scores.

    Args:
        input_file (str): Path to the input JSONL file

    Returns:
        tuple: List of data items and array of IDF scores
    """
    idf_scores = []
    data_items = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line)
            data_items.append(data)
            idf = data['idf']
            idf_scores.append(idf)

    return data_items, np.array(idf_scores).reshape(-1, 1)


def assign_dataset_task_ids(data_items):
    """
    Assign a unique integer task ID to each unique dataset name.

    Args:
        data_items (list): List of data items

    Returns:
        dict: Mapping from dataset names to integer task IDs
    """
    dataset_names = set()
    for item in data_items:
        if 'dataset' in item:
            dataset_names.add(item['dataset'])

    dataset_to_task_id = {name: i for i, name in enumerate(sorted(dataset_names))}

    print(f"Found {len(dataset_to_task_id)} unique datasets with assigned task IDs:")
    for name, task_id in dataset_to_task_id.items():
        print(f"  {name}: {task_id}")

    return dataset_to_task_id


def perform_clustering(X, n_clusters=7, random_state=42):
    """
    Perform K-means clustering on the data.

    Args:
        X (numpy.ndarray): Input data for clustering
        n_clusters (int): Number of clusters to form
        random_state (int): Random state for reproducibility

    Returns:
        tuple: Cluster labels and centers
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans.labels_, kmeans.cluster_centers_


def print_cluster_info(labels, centers):
    """
    Print information about the clusters.

    Args:
        labels (numpy.ndarray): Cluster labels
        centers (numpy.ndarray): Cluster centers
    """
    print("Cluster centers:")
    for i, center in enumerate(centers):
        print(f"  Cluster {i}: {center[0]:.4f}")

    cluster_counts = Counter(labels)
    print("\nCluster sample counts:")
    for cluster, count in sorted(cluster_counts.items()):
        print(f"  Cluster {cluster}: {count} samples")


def save_clustered_data(data_items, labels, dataset_to_task_id, output_file):
    """
    Save the original data with added cluster labels and task IDs.

    Args:
        data_items (list): Original data items
        labels (numpy.ndarray): Cluster labels
        dataset_to_task_id (dict): Mapping from dataset names to integer task IDs
        output_file (str): Path to the output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, data in enumerate(data_items):
            # Add cluster label
            data['idf_cluster'] = int(labels[i])

            # Add task ID if dataset field exists
            if 'dataset' in data:
                dataset_name = data['dataset']
                data['task'] = dataset_to_task_id.get(dataset_name, -1)  # Default to -1 if not found

            f.write(json.dumps(data) + '\n')

    print(f"Saved clustered data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Perform K-means clustering on IDF scores and assign task IDs')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file path')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file path')
    parser.add_argument('--clusters', type=int, default=7, help='Number of clusters (default: 7)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random state (default: 42)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load data
    print(f"Loading data from {args.input}...")
    data_items, X = load_data(args.input)
    print(f"Loaded {len(data_items)} items")

    # Assign task IDs based on dataset names
    print("Assigning unique task IDs to datasets...")
    dataset_to_task_id = assign_dataset_task_ids(data_items)

    # Perform clustering
    print(f"Performing K-means clustering with {args.clusters} clusters...")
    labels, centers = perform_clustering(X, n_clusters=args.clusters, random_state=args.random_state)

    # Print cluster information
    print_cluster_info(labels, centers)

    # Save clustered data with task IDs
    save_clustered_data(data_items, labels, dataset_to_task_id, args.output)


if __name__ == "__main__":
    main()