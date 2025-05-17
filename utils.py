import numpy as np
import tensorflow as tf

# Utility to generate sample text from model
def Sample_text(model, start_string, length=500, temperature=1.0):
    # Load character to index mapping
    try:
        with open('char_to_idx.pkl', 'rb') as f:
            char_to_idx = pickle.load(f)
        with open('idx_to_char.pkl', 'rb') as f:
            idx_to_char = pickle.load(f)
    except FileNotFoundError:
        return "[char_to_idx.pkl or idx_to_char.pkl not found]"

    input_indices = [char_to_idx.get(s, 0) for s in start_string]
    input_tensor = tf.expand_dims(input_indices, 0)  # batch size = 1
    generated = []

    model.reset_states()

    for _ in range(length):
        predictions = model(input_tensor)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_tensor = tf.expand_dims([predicted_id], 0)
        generated.append(idx_to_char[predicted_id])

    return start_string + ''.join(generated)
