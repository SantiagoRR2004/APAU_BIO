import numpy as np
import tensorflow as tf


def timeseries_dataset_from_array(
    data,
    targets,
    sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
):
    start_index = start_index or 0
    end_index = end_index or len(data)

    # Apply sampling rate
    data = data[start_index:end_index:sampling_rate]
    targets = targets[start_index:end_index:sampling_rate]

    # Calculate total number of sequences
    num_sequences = (len(data) - sequence_length) // sequence_stride + 1

    # Generate sequence start indices
    indices = np.arange(0, num_sequences * sequence_stride, sequence_stride)

    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)

    # Create all sequences and corresponding targets
    sequences = np.zeros((len(indices), sequence_length, data.shape[1]))
    sequence_targets = np.zeros((len(indices),))

    for i, idx in enumerate(indices):
        start = idx
        end = start + sequence_length
        sequences[i] = data[start:end]
        sequence_targets[i] = targets[end - 1]

    return tf.data.Dataset.from_tensor_slices((sequences, sequence_targets)).batch(
        batch_size
    )
