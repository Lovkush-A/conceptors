import torch
from typing import List
from tqdm import tqdm


def extract_activations_last_token(model, steering_prompts, extraction_layers, device):
    """
    Extract activations for the last token of each steering prompt from specific layers of the model.

    Parameters:
    - model (HookedTransformer): The model used for generating text.
    - steering_prompts (list): List of steering prompts to extract activations for.
    - extraction_layers (list): The layers from which activations are extracted.
    - device (str): The computing device (e.g., 'cuda', 'cpu').

    Returns:
    - dict: A dictionary where each key is a layer number and each value is the
            activations for the last token of each prompt. Shape: (n_prompts, n_activations).
    """
    activations_dict = {}
    names = [f"blocks.{layer}.hook_resid_pre" for layer in extraction_layers]
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n in names)

    with model.hooks(fwd_hooks=caching_hooks):
        model.tokenizer.padding_side = "left"
        _ = model(steering_prompts)

    for layer in extraction_layers:
        prompt_activations = cache[f"blocks.{layer}.hook_resid_pre"].detach().cpu()
        last_token_activations = prompt_activations[:, -1, :].squeeze()
        activations_tensor = torch.tensor(
            last_token_activations.numpy(), dtype=torch.float, device='cpu'
        )
        activations_dict[layer] = activations_tensor

    return activations_dict


def generate_activations(
    model,
    experiment_data: dict,
    list_extraction_layers: List[int],
    num_experiments: int,
    device: str = "cuda",
):
    # Initialize a dictionary to store activations for each experiment and layer
    activations_cache = {exp: {} for exp in range(num_experiments)}

    # Precompute activations for all layers with a progress bar
    total_computations = num_experiments

    with tqdm(total=total_computations, desc="Precomputing activations") as pbar:
        for exp in range(num_experiments):
            activations_dict = extract_activations_last_token(
                model,
                experiment_data[exp]["steering_examples"],
                list_extraction_layers,
                device=device,
            )
            for layer in list_extraction_layers:
                activations_cache[exp][layer] = activations_dict[layer].squeeze()
            pbar.update(1)
    
    return activations_cache


def steer(C, x, beta):
    """
    Steers the given vector x using the conceptor C.

    Args:
        C (torch.Tensor): The conceptor matrix.
        x (torch.Tensor): The vector to be steered.
        beta (float): The steering parameter with 0: no steering, 1: full steering.

    Returns:
        torch.Tensor: The steered vector.
    """
    C = C.to(torch.float16)
    return beta * torch.matmul(C, x)


def top_1_first_tokens(model, prompts: List[str], fwd_hooks=[]):
    """
    Retrieves the top token predictions for the first tokens after the prompts.

    Parameters:
    - model: Language model with hooks and tokenizer.
    - prompts (List[str]): List of prompt strings.
    - fwd_hooks (list, optional): List of forward hooks to apply during the model run.

    Returns:
    - List[str]: List of top predicted tokens.
    """
    top_tokens = []

    with model.hooks(fwd_hooks=fwd_hooks):
        model.tokenizer.padding_side = "left"
        input_prompts_tokenized = model.to_tokens(prompts)
        logits, _ = model.run_with_cache(input_prompts_tokenized, remove_batch_dim=True)
        next_logits = logits[:, -1, :]
        next_probabilities = next_logits.softmax(dim=-1)
        top_indices = torch.argmax(next_probabilities, dim=-1)

        for index in top_indices:
            decoded_token = model.tokenizer.decode([index.item()])
            top_tokens.append(decoded_token)

    return top_tokens


def generate_conceptor_hook(conceptor, beta):
    """
    Generates a hook function to apply a conceptor to the last token.

    Parameters:
    - conceptor (torch.Tensor): Conceptor matrix.
    - beta (float): Scaling factor.

    Returns:
    - function: Hook function for applying conceptor.
    """

    def last_token_steering_hook(resid_pre, hook):
        for i in range(resid_pre.shape[0]):
            current_token_index = resid_pre.shape[1] - 1
            resid_pre[i, current_token_index, :] = steer(
                C=conceptor, x=resid_pre[i, current_token_index, :], beta=beta
            )

    return last_token_steering_hook


def generate_conceptor_hook_mean_centered(conceptor, mean_train, beta):
    """
    Generates a hook function to apply a mean-centered conceptor to the last token.

    Parameters:
    - conceptor (torch.Tensor): Conceptor matrix.
    - mean_train (torch.Tensor): Mean training vector.
    - beta (float): Scaling factor.

    Returns:
    - function: Hook function for applying mean-centered conceptor.
    """

    def last_token_steering_hook(resid_pre, hook):
        for i in range(resid_pre.shape[0]):
            current_token_index = resid_pre.shape[1] - 1
            resid_pre[i, current_token_index, :] = (
                steer(
                    C=conceptor,
                    x=resid_pre[i, current_token_index, :] - mean_train,
                    beta=beta,
                )
                + mean_train
            )

    return last_token_steering_hook


def generate_ave_hook_addition(steering_vector, beta):
    """
    Generates a hook function to add a steering vector to the last token.

    Parameters:
    - steering_vector (torch.Tensor): Steering vector.
    - beta (float): Scaling factor.

    Returns:
    - function: Hook function for adding steering vector.
    """

    def last_token_steering_hook(resid_pre, hook):
        for i in range(resid_pre.shape[0]):
            current_token_index = resid_pre.shape[1] - 1
            resid_pre[i, current_token_index, :] += steering_vector.squeeze().to(resid_pre.device) * beta

    return last_token_steering_hook


def generate_ave_hook_addition_mean_centered(steering_vector, mean_train, beta):
    """
    Generates a hook function to add a mean-centered steering vector to the last token.

    Parameters:
    - steering_vector (torch.Tensor): Steering vector.
    - mean_train (torch.Tensor): Mean training vector.
    - beta (float): Scaling factor.

    Returns:
    - function: Hook function for adding mean-centered steering vector.
    """

    def last_token_steering_hook(resid_pre, hook):
        for i in range(resid_pre.shape[0]):
            current_token_index = resid_pre.shape[1] - 1
            resid_pre[i, current_token_index, :] += (
                steering_vector.squeeze().to(resid_pre.device) - mean_train.to(resid_pre.device)
            ) * beta

    return last_token_steering_hook