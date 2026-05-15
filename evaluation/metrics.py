"""
Evaluation metrics for substitution cipher KEY prediction.

NEW APPROACH: Metrics focused on key accuracy and decryption quality.

Implements:
- Key accuracy (% of 26 letters correct)
- Perfect key recovery rate (all 26 letters correct)
- Character accuracy (apply key to decrypt, compare to plaintext)
- Word accuracy (after applying key)
"""


def key_accuracy(predicted_key, true_key):
    """
    Calculate percentage of key letters correctly predicted.

    Args:
        predicted_key: Predicted 26-letter key string
        true_key: True 26-letter key string

    Returns:
        Accuracy as percentage (0-100)
    """
    if len(predicted_key) != 26 or len(true_key) != 26:
        return 0.0

    correct = sum(1 for i in range(26) if predicted_key[i] == true_key[i])
    return 100.0 * correct / 26


def perfect_key_recovery(predicted_key, true_key):
    """
    Check if all 26 letters of the key are correctly predicted.

    Args:
        predicted_key: Predicted 26-letter key string
        true_key: True 26-letter key string

    Returns:
        1.0 if perfect match, 0.0 otherwise
    """
    return 1.0 if predicted_key == true_key else 0.0


def apply_key(ciphertext, key_string):
    """
    Apply substitution key to decrypt ciphertext.

    Args:
        ciphertext: Encrypted text string
        key_string: 26-character key where key_string[i] is the plaintext
                   letter that cipher letter chr(ord('a') + i) decrypts to

    Returns:
        Decrypted plaintext string
    """
    if len(key_string) != 26:
        return ""

    # Build decryption mapping
    decryption_map = {}
    for i in range(26):
        cipher_char = chr(ord('a') + i)
        plain_char = key_string[i]
        decryption_map[cipher_char] = plain_char

    # Decrypt
    plaintext = ''.join(decryption_map.get(char, char) for char in ciphertext)
    return plaintext


def character_accuracy(predicted_plaintext, true_plaintext):
    """
    Calculate character-level accuracy after decryption.

    Args:
        predicted_plaintext: Decrypted text using predicted key
        true_plaintext: True plaintext

    Returns:
        Accuracy as percentage (0-100)
    """
    if len(true_plaintext) == 0:
        return 100.0 if len(predicted_plaintext) == 0 else 0.0

    # Ensure same length for comparison
    min_len = min(len(predicted_plaintext), len(true_plaintext))
    max_len = max(len(predicted_plaintext), len(true_plaintext))

    # Count matching characters
    correct = sum(1 for i in range(min_len) if predicted_plaintext[i] == true_plaintext[i])

    # Accuracy as percentage
    accuracy = 100.0 * correct / max_len

    return accuracy


def word_accuracy(predicted_plaintext, true_plaintext):
    """
    Calculate word-level accuracy after decryption.

    Args:
        predicted_plaintext: Decrypted text using predicted key
        true_plaintext: True plaintext

    Returns:
        Accuracy as percentage (0-100)
    """
    pred_words = predicted_plaintext.split()
    true_words = true_plaintext.split()

    if len(true_words) == 0:
        return 100.0 if len(pred_words) == 0 else 0.0

    # Count correctly predicted words
    min_len = min(len(pred_words), len(true_words))
    correct_words = sum(1 for i in range(min_len) if pred_words[i] == true_words[i])

    # Accuracy based on total target words
    accuracy = 100.0 * correct_words / len(true_words)

    return accuracy


def calculate_all_metrics(predicted_key, true_key, ciphertext, true_plaintext):
    """
    Calculate all evaluation metrics for key prediction.

    Args:
        predicted_key: Predicted 26-letter key string
        true_key: True 26-letter key string
        ciphertext: Encrypted text
        true_plaintext: True plaintext

    Returns:
        Dictionary with all metrics
    """
    # Key metrics
    key_acc = key_accuracy(predicted_key, true_key)
    perfect_key = perfect_key_recovery(predicted_key, true_key)

    # Apply predicted key to decrypt
    predicted_plaintext = apply_key(ciphertext, predicted_key)

    # Decryption quality metrics
    char_acc = character_accuracy(predicted_plaintext, true_plaintext)
    word_acc = word_accuracy(predicted_plaintext, true_plaintext)

    return {
        'key_accuracy': key_acc,
        'perfect_key_recovery': perfect_key,
        'character_accuracy': char_acc,
        'word_accuracy': word_acc,
        'predicted_plaintext': predicted_plaintext
    }


def aggregate_metrics(all_metrics):
    """
    Calculate aggregate metrics across multiple samples.

    Args:
        all_metrics: List of metric dictionaries from calculate_all_metrics

    Returns:
        Dictionary with averaged metrics
    """
    if len(all_metrics) == 0:
        return {
            'key_accuracy': 0.0,
            'perfect_key_recovery_rate': 0.0,
            'character_accuracy': 0.0,
            'word_accuracy': 0.0
        }

    n = len(all_metrics)

    return {
        'key_accuracy': sum(m['key_accuracy'] for m in all_metrics) / n,
        'perfect_key_recovery_rate': sum(m['perfect_key_recovery'] for m in all_metrics) / n * 100,
        'character_accuracy': sum(m['character_accuracy'] for m in all_metrics) / n,
        'word_accuracy': sum(m['word_accuracy'] for m in all_metrics) / n
    }


def test_metrics():
    """
    Test function for evaluation metrics.
    """
    print("Testing key prediction metrics:")
    print("="*80)

    # Test case 1: Perfect key prediction
    true_key = "bcdefghijklmnopqrstuvwxyza"
    predicted_key = "bcdefghijklmnopqrstuvwxyza"
    ciphertext = "ifmmp xpsme"
    true_plaintext = "hello world"

    print("\nTest 1: Perfect key prediction")
    print(f"True key:      {true_key}")
    print(f"Predicted key: {predicted_key}")
    print(f"Ciphertext:    {ciphertext}")
    print(f"True plaintext: {true_plaintext}")

    metrics = calculate_all_metrics(predicted_key, true_key, ciphertext, true_plaintext)
    print(f"Key accuracy:          {metrics['key_accuracy']:.2f}%")
    print(f"Perfect key recovery:  {metrics['perfect_key_recovery']}")
    print(f"Character accuracy:    {metrics['character_accuracy']:.2f}%")
    print(f"Word accuracy:         {metrics['word_accuracy']:.2f}%")
    print(f"Decrypted:             {metrics['predicted_plaintext']}")

    # Test case 2: Partial key prediction (23/26 correct)
    predicted_key_partial = "bcdefghijklmnopqrstuvwxyzz"  # Last letter wrong

    print("\nTest 2: Partial key prediction (25/26 correct)")
    print(f"True key:      {true_key}")
    print(f"Predicted key: {predicted_key_partial}")

    metrics2 = calculate_all_metrics(predicted_key_partial, true_key, ciphertext, true_plaintext)
    print(f"Key accuracy:          {metrics2['key_accuracy']:.2f}%")
    print(f"Perfect key recovery:  {metrics2['perfect_key_recovery']}")
    print(f"Character accuracy:    {metrics2['character_accuracy']:.2f}%")
    print(f"Decrypted:             {metrics2['predicted_plaintext']}")

    # Test case 3: Aggregation
    print("\nTest 3: Aggregate metrics")
    all_metrics = [metrics, metrics2]
    agg = aggregate_metrics(all_metrics)
    print(f"Average key accuracy:         {agg['key_accuracy']:.2f}%")
    print(f"Perfect key recovery rate:    {agg['perfect_key_recovery_rate']:.2f}%")
    print(f"Average character accuracy:   {agg['character_accuracy']:.2f}%")


if __name__ == '__main__':
    test_metrics()
