import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, List, Optional


def cache_activation_vectors(
    activations_cache: Dict[int, Dict[int, np.ndarray]],
    list_extraction_layers: List[int],
    num_experiments: int,
) -> Dict[int, Dict[int, torch.Tensor]]:
    """
    Precomputes averaged activations for all layers and experiments.

    Args:
        activations_cache: dictionary containing the activations for each experiment and layer.
            Keys are (experiment_index) then (layer_index) and values are the activations.
            Shape of values: (NUM_STEERING_PROMPTS, d_model).
            (NUM_STEERING_PROMPTS is number of prompts used to train the steering vector per experiment.)
        list_extraction_layers: list of layers to extract activations from.
            Redundant, as it should be in activations_cache.
        num_experiments: number of experiments to precompute activations for.
            Redundant, as it should be in activations_cache.

    Returns:
        averaged_activations_cache: dictionary containing the averaged activations for each experiment and layer.
            Keys are (experiment_index) then (layer_index) and values are the averaged activations.
            Shape of values: (d_model,).
    """
    # Initialize a dictionary to store averaged activations for each experiment
    averaged_activations_cache = {exp: {} for exp in range(num_experiments)}

    # Total number of computations for the progress bar
    total_computations = len(list_extraction_layers) * num_experiments

    # Precompute averaged activations for all layers with a progress bar
    with tqdm(total=total_computations, desc="Precomputing averaged activations") as pbar:
        for exp in range(num_experiments):
            for layer in list_extraction_layers:
                # Extract the last-token activations of steering examples at the specified layer
                activations = activations_cache[exp][layer]
                # Compute the average activations
                avg_activations = torch.mean(activations, dim=0)
                # Store the average activations in the cache
                averaged_activations_cache[exp][layer] = avg_activations.detach().cpu()
                # Update the progress bar
                pbar.update(1)

    return averaged_activations_cache
