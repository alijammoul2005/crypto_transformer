"""
Evaluation metrics for substitution cipher decryption.

Implements:
- Character accuracy
- Word accuracy
- BLEU score
- Mean edit distance (Levenshtein distance)
"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein


def character_accuracy(predicted, target):
    """
    Calculate character-level accuracy.

    Measures percentage of characters correctly predicted.

    Args:
        predicted: Predicted plaintext string
        target: True plaintext string

    Returns:
        Accuracy as percentage (0-100)
    """
    if len(target) == 0:
        return 100.0 if len(predicted) == 0 else 0.0

    # Ensure same length for comparison
    min_len = min(len(predicted), len(target))
    max_len = max(len(predicted), len(target))

    # Count matching characters
    correct = sum(1 for i in range(min_len) if predicted[i] == target[i])

    # Accuracy as percentage
    accuracy = 100.0 * correct / max_len

    return accuracy


def word_accuracy(predicted, target):
    """
    Calculate word-level accuracy.

    Measures percentage of complete words correctly recovered.

    Args:
        predicted: Predicted plaintext string
        target: True plaintext string

    Returns:
        Accuracy as percentage (0-100)
    """
    pred_words = predicted.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 100.0 if len(pred_words) == 0 else 0.0

    # Count correctly predicted words
    # Align by position for fair comparison
    min_len = min(len(pred_words), len(target_words))
    correct_words = sum(1 for i in range(min_len) if pred_words[i] == target_words[i])

    # Accuracy based on total target words
    accuracy = 100.0 * correct_words / len(target_words)

    return accuracy


def bleu_score(predicted, target):
    """
    Calculate BLEU score.

    Uses NLTK's sentence_bleu with smoothing for short sequences.

    Args:
        predicted: Predicted plaintext string
        target: True plaintext string

    Returns:
        BLEU score (0-1)
    """
    # Tokenize by characters for character-level BLEU
    pred_tokens = list(predicted)
    target_tokens = list(target)

    if len(target_tokens) == 0:
        return 1.0 if len(pred_tokens) == 0 else 0.0

    # Use smoothing function for short sequences
    smoothing = SmoothingFunction().method1

    # Calculate BLEU score
    # target should be a list of reference translations
    score = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothing)

    return score


def mean_edit_distance(predicted, target):
    """
    Calculate Levenshtein edit distance.

    Measures minimum number of single-character edits needed to transform
    predicted string into target string.

    Args:
        predicted: Predicted plaintext string
        target: True plaintext string

    Returns:
        Edit distance (integer)
    """
    return Levenshtein.distance(predicted, target)


def calculate_all_metrics(predicted, target):
    """
    Calculate all evaluation metrics.

    Args:
        predicted: Predicted plaintext string
        target: True plaintext string

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'character_accuracy': character_accuracy(predicted, target),
        'word_accuracy': word_accuracy(predicted, target),
        'bleu_score': bleu_score(predicted, target),
        'edit_distance': mean_edit_distance(predicted, target)
    }

    return metrics


def aggregate_metrics(all_predictions, all_targets):
    """
    Calculate aggregate metrics across multiple samples.

    Args:
        all_predictions: List of predicted plaintext strings
        all_targets: List of true plaintext strings

    Returns:
        Dictionary with averaged metrics
    """
    if len(all_predictions) != len(all_targets):
        raise ValueError("Number of predictions must match number of targets")

    if len(all_predictions) == 0:
        return {
            'character_accuracy': 0.0,
            'word_accuracy': 0.0,
            'bleu_score': 0.0,
            'mean_edit_distance': 0.0
        }

    total_char_acc = 0.0
    total_word_acc = 0.0
    total_bleu = 0.0
    total_edit_dist = 0.0

    for pred, target in zip(all_predictions, all_targets):
        metrics = calculate_all_metrics(pred, target)
        total_char_acc += metrics['character_accuracy']
        total_word_acc += metrics['word_accuracy']
        total_bleu += metrics['bleu_score']
        total_edit_dist += metrics['edit_distance']

    n = len(all_predictions)

    return {
        'character_accuracy': total_char_acc / n,
        'word_accuracy': total_word_acc / n,
        'bleu_score': total_bleu / n,
        'mean_edit_distance': total_edit_dist / n
    }


def test_metrics():
    """
    Test function for evaluation metrics.
    """
    # Test cases
    test_cases = [
        ("hello world", "hello world", "Perfect match"),
        ("hello world", "hello worlx", "One character off"),
        ("hello world", "hallo world", "One substitution"),
        ("hello world", "world hello", "Word order changed"),
        ("", "", "Both empty"),
        ("hello", "hello there", "Target longer"),
    ]

    print("Testing evaluation metrics:")
    print("="*80)

    for pred, target, description in test_cases:
        print(f"\n{description}:")
        print(f"  Predicted: '{pred}'")
        print(f"  Target:    '{target}'")

        metrics = calculate_all_metrics(pred, target)

        print(f"  Character Accuracy: {metrics['character_accuracy']:.2f}%")
        print(f"  Word Accuracy:      {metrics['word_accuracy']:.2f}%")
        print(f"  BLEU Score:         {metrics['bleu_score']:.4f}")
        print(f"  Edit Distance:      {metrics['edit_distance']}")


if __name__ == '__main__':
    test_metrics()
