import pandas as pd
from sklearn.model_selection import train_test_split

path_to_data = "antonyms.json"

def load_and_split_data(
    path_to_data: str,
    split_ratio: float = 0.8,
    random_seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and splits data into training and testing sets.

    Args:
        path_to_data (str): The path to the data file.
        split_ratio (float): The ratio of the data to split into training and testing sets.
        random_seed (int): The random seed to use for the split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing sets.
    """
    data = pd.read_json(path_to_data)
    train_data, test_data = train_test_split(data, test_size=split_ratio, random_state=random_seed)
    return train_data, test_data


def generate_training_prompts(
    data: pd.DataFrame,
    n_pairs_per_training_prompt: int = 10,
    separator: str = ', '
) -> list[str]:
    r"""
    Generates training prompts from the data.

    Args:
        data (pd.DataFrame): The data to generate training prompts from.
            Must have 'input' and 'target' columns.
        n_pairs_per_training_prompt (int): The number of pairs per training prompt.
        separator (str): The separator to use between pairs.
    Returns:
        list[str]: A list of training prompts.
            Each training prompt is a string of the form:
            "input1:target1\ninput2:target2\n...\ninputN:"
            In particular, the last target is not included, and it ends on a colon.
    """
    n_training_prompts = len(data) // n_pairs_per_training_prompt

    if n_training_prompts == 0:
        raise ValueError("Not enough data to generate training prompts.")

    training_prompts: list[str] = []
    for i in range(n_training_prompts):
        training_prompt = ""
        for j in range(n_pairs_per_training_prompt-1):
            training_prompt += f"{data.iloc[i*n_pairs_per_training_prompt + j]['input']}"
            training_prompt += ':'
            training_prompt += f"{data.iloc[i*n_pairs_per_training_prompt + j]['target']}"
            training_prompt += separator
        training_prompt += f"{data.iloc[i*n_pairs_per_training_prompt + n_pairs_per_training_prompt-1]['input']}"
        training_prompt += ':'
        training_prompts.append(training_prompt)
    
    return training_prompts