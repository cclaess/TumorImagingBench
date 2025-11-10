import logging
import numpy as np
import optuna
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import plotly.express as px


def train_knn_classifier(train_items, train_labels, val_items, val_labels):
    """
    Train a KNN classifier with hyperparameter optimization using Optuna.

    Args:
        train_items: Training feature matrix.
        train_labels: Training labels.
        val_items: Validation feature matrix.
        val_labels: Validation labels.

    Returns:
        best_model: Trained KNeighborsClassifier with the best parameters.
        study: Optuna study object with the optimization results.
    """
    def objective(trial):
        k = trial.suggest_int('k', 1, 50)
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(train_items, train_labels)
        val_predictions = knn.predict_proba(val_items)
        if val_predictions.shape[1] == 2:  # Binary classification
            return roc_auc_score(val_labels, val_predictions[:, 1])
        else:
            return roc_auc_score(val_labels, val_predictions, multi_class='ovr')

    # Define grid of k values from 1 to 50
    param_grid = {'k': list(range(1, 51))}
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.GridSampler(param_grid)
    )
    study.optimize(objective, n_trials=len(param_grid['k']))

    # Train the final model using the best hyperparameter
    best_model = KNeighborsClassifier(n_neighbors=study.best_params['k'], metric='cosine')
    best_model.fit(train_items, train_labels)

    return best_model, study


def evaluate_model(model, test_items, test_labels):
    """
    Evaluate a trained classifier on test data using the ROC AUC score.

    Args:
        model: Trained classifier.
        test_items: Test feature matrix.
        test_labels: Test labels.

    Returns:
        ROC AUC score as a float.
    """
    test_predictions = model.predict_proba(test_items)
    if test_predictions.shape[1] == 2:  # Binary classification
        return roc_auc_score(test_labels, test_predictions[:, 1])
    else:
        return roc_auc_score(test_labels, test_predictions, multi_class='ovr')


def plot_model_comparison(test_accuracies_dict, width=500, height=400, font_size=20, marker_color="#ADD8E6", yshift_annotation=20):
    """
    Create a minimalist and elegant bar plot comparing model performances using Plotly Express.

    Args:
        test_accuracies_dict: Dictionary mapping model names to their performance metrics.
            Each value should be a dict with keys 'mean' and 'ci_95', where 'ci_95' is a tuple (lower_bound, upper_bound).

    Returns:
        Plotly figure object.
    """
    # Extract model names, mean values, and compute error bars for the 95% CI.
    model_names = list(test_accuracies_dict.keys())
    means = [test_accuracies_dict[model]['mean'] for model in model_names]
    error_y = [test_accuracies_dict[model]['ci95'][1] - test_accuracies_dict[model]['mean'] for model in model_names]
    error_y_minus = [test_accuracies_dict[model]['mean'] - test_accuracies_dict[model]['ci95'][0] for model in model_names]

    fig = px.bar(
        x=model_names,
        y=means,
        error_y=error_y,
        error_y_minus=error_y_minus,
        labels={'x': 'Model', 'y': 'AUC'},
        title='',
        template='simple_white',
        width=width,
        height=height
    )
    # Use a subtle blue color and position text inside each bar with minimal formatting;
    # update error bar thickness to make them thicker.
    fig.update_traces(
        marker_color=marker_color,
        marker_opacity=0.8,
        error_y=dict(width=10)
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showline=False, showgrid=False, tickfont=dict(size=font_size)),
        yaxis=dict(showgrid=False, tickfont=dict(size=font_size)),
        title=dict(x=0.5, xanchor='center'),
        font=dict(size=font_size, family="Arial"),
        showlegend=False
    )
    
    # Add annotations for mean scores positioned just above the upper confidence interval.
    for model, mean, err in zip(model_names, means, error_y):
        fig.add_annotation(
            x=model,
            y=mean + err,
            text=f"{mean:.2f}",
            showarrow=False,
            yshift=yshift_annotation,
            font=dict(size=font_size - 2, color="black")
        )
        
    return fig


def split_shuffle_data(items, labels, train_ratio=0.5, val_ratio=0.2, random_seed=42, stratify=False):
    """
    Split and shuffle data into training, validation, and test sets.

    Args:
        items: Feature array.
        labels: Label array.
        train_ratio: Ratio of data used for training.
        val_ratio: Ratio of data used for validation.
        random_seed: Random seed for reproducibility.
        stratify: If True, stratify splits based on label distribution.

    Returns:
        Tuple containing (train_items, train_labels, val_items, val_labels, test_items, test_labels).
    """
    if stratify:
        # First, split off the test set
        test_ratio = 1 - train_ratio - val_ratio
        train_val_items, test_items, train_val_labels, test_labels = train_test_split(
            items, labels,
            test_size=test_ratio,
            random_state=random_seed,
            stratify=labels
        )
        # Then, split the remaining data into training and validation sets
        train_items, val_items, train_labels, val_labels = train_test_split(
            train_val_items, train_val_labels,
            test_size=val_ratio / (train_ratio + val_ratio),
            random_state=random_seed,
            stratify=train_val_labels
        )
    else:
        rng = np.random.default_rng(random_seed)
        shuffled_indices = rng.permutation(len(labels))
        items = items[shuffled_indices]
        labels = np.array(labels)[shuffled_indices]

        train_size = int(train_ratio * len(labels))
        val_size = int(val_ratio * len(labels))

        train_items = items[:train_size]
        train_labels = labels[:train_size]
        val_items = items[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        test_items = items[train_size + val_size:]
        test_labels = labels[train_size + val_size:]
    return train_items, train_labels, val_items, val_labels, test_items, test_labels


def apply_aggregation_filter(v, model_name):
    if model_name == "MedImageInsightExtractor":
        return v.mean(axis=0)
    elif model_name == "CTClipVitExtractor":
        return v.mean(axis=(1,2,3))
    elif model_name == "PASTAExtractor":
        return v.mean(axis=(2,3,4))   
    elif model_name == "SpectreLocalViTExtractor":
        # cls = v[:, :, 0].mean(axis=1)
        # pathch = v[:, :, 1:].mean(axis=(1,2))
        return v[:, :, 0].mean(axis=1)
    else:
        return v

def extract_model_features(data):
    """
    Concatenate features from the train, validation, and test sets for each model.

    Args:
        data (dict): Dictionary where each key is a model name and each value is a dict 
                     with lists for 'train', 'val', and 'test'. Each list contains 
                     dictionaries with a "feature" key.
        skip_model (str): Model name to skip during feature extraction.

    Returns:
        Dictionary mapping model names to a concatenated numpy array of features.
    """
    model_features = {}
    for model_name, splits in data.items():
        features_to_concat = []
        for split in ["train", "val", "test"]:
            if split in splits and splits[split]:
                # Stack features for this split and add to the list.
                split_features = np.vstack([apply_aggregation_filter(entry["feature"], model_name) for entry in splits[split]])
                features_to_concat.append(split_features)
        # Concatenate all split features along axis 0 if any exist; otherwise, use an empty array.
        model_features[model_name] = np.concatenate(features_to_concat, axis=0) if features_to_concat else np.array([])
    return model_features


def compute_knn_indices(model_features, num_neighbors=10, metric="cosine"):
    """
    Compute k-nearest neighbor indices (excluding the sample itself) for each model's features.

    Args:
        model_features (dict): Dictionary mapping model names to feature arrays.
        num_neighbors (int): Number of nearest neighbors to retrieve (excluding self).
        metric (str): Distance metric to use.

    Returns:
        Dictionary mapping model names to an array of nearest neighbor indices.
    """
    model_neighbors = {}
    for model_name, features in model_features.items():
        nn_model = NearestNeighbors(n_neighbors=num_neighbors + 1, metric=metric)
        nn_model.fit(features)
        _, indices = nn_model.kneighbors(features)
        model_neighbors[model_name] = indices[:, 1:]  # Exclude self-neighbor
    return model_neighbors


def compute_overlap_matrix(model_neighbors):
    """
    Compute average mutual k-nearest neighbor overlap scores between pairs of models.

    For each pair of models, this function calculates the average overlap score based on mutual nearest neighbors.
    If the number of samples between models does not match, a warning is issued and that pair is skipped.

    Args:
        model_neighbors (dict): Dictionary mapping model names to arrays of neighbor indices.

    Returns:
        A tuple (overlap_matrix, model_list), where overlap_matrix is a symmetric numpy array of
        average overlap scores and model_list is a list of corresponding model names.
    """
    model_list = list(model_neighbors.keys())
    n_models = len(model_list)
    overlap_matrix = np.full((n_models, n_models), np.nan)

    for i in range(n_models):
        neighbors_a = model_neighbors[model_list[i]]
        for j in range(i + 1, n_models):
            neighbors_b = model_neighbors[model_list[j]]
            if neighbors_a.shape[0] != neighbors_b.shape[0]:
                logging.warning(
                    "Number of samples in %s and %s do not match. Skipping pair.",
                    model_list[i], model_list[j]
                )
                continue

            # Determine overlap using broadcasting
            common_flags = (neighbors_a[:, :, None] == neighbors_b[:, None, :]).any(axis=2)
            sample_overlaps = np.sum(common_flags, axis=1)
            avg_overlap = np.mean(sample_overlaps)

            overlap_matrix[i, j] = avg_overlap
            overlap_matrix[j, i] = avg_overlap
    return overlap_matrix, model_list


def plot_overlap_matrix(overlap_matrix, model_list, title="Mutual k-Nearest Neighbors Overlap Scores", width=1200, height=1200, color="Greens", tickangle=90, font_size=30):
    """
    Plot the mutual k-nearest neighbor overlap matrix using Plotly Express.

    Args:
        overlap_matrix (numpy.ndarray): Square matrix with average overlap scores.
        model_list (list): List of model names corresponding to the matrix axes.
        title (str): Title of the plot.
        width (int): Width of the plot.
        tickangle (int): Angle for x-axis tick labels.

    Returns:
        Plotly figure object.
    """
    fig = px.imshow(
        overlap_matrix,
        labels={"x": "Model", "y": "Model", "color": "Overlap"},
        x=model_list,
        y=model_list,
        color_continuous_scale=color
    )
    fig.update_layout(
        title="",
        width=width,
        height=height,
        font=dict(size=font_size, family="Arial"),
        xaxis_tickangle=tickangle,
        template="simple_white",
        coloraxis_colorbar=dict(
        yanchor="bottom",
        y=1.02,  # Position slightly above the plot
        x=0.5,   # Center horizontally
        xanchor="center",
        orientation="h"  # Horizontal orientation
    )
    )
    return fig
