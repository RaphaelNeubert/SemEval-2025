
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_predictions(ground_truth_file, predictions_file):
    # Load ground truth and predictions
    ground_truth = pd.read_csv(ground_truth_file)
    predictions = pd.read_csv(predictions_file)

    # Remove non-label columns (assuming first two columns are ID and text in ground truth)
    label_columns = ground_truth.columns[2:]  # Skip 'id' and 'text' columns

    # Ensure prediction file contains the same columns as ground truth (excluding 'text')
    if not set(label_columns).issubset(set(predictions.columns)):
        raise ValueError("Prediction file must contain the same label columns as the ground truth.")

    # Extract the labels from both files
    y_true = ground_truth[label_columns].values
    y_pred = predictions[label_columns].values

    # Compute accuracy (exact match)
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())

    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Micro Precision: {precision:.4f}, Micro Recall: {recall:.4f}, Micro F1-score: {f1:.4f}")
    print(f"Macro Precision: {precision_macro:.4f}, Macro Recall: {recall_macro:.4f}, Macro F1-score: {f1_macro:.4f}")

# Example usage
ground_truth_file = 'data/public_data_test/track_a/dev/eng.csv'
predictions_file = 'data/pred_eng.csv'

evaluate_predictions(ground_truth_file, predictions_file)
