import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras


# --------------------------------
# Carga de datos
# --------------------------------

fname = os.path.join("/home/leandro/datasets/jena_climate_2009_2016.csv")

with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print("Campos:", header)
print("Número de registros:", len(lines))

temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]  # Column 1 is the temperature array
    raw_data[i, :] = values[
        :
    ]  # All columns (including temperature) is the "raw_data" array
print("Valores temperatura:", temperature)
print("Dimensión valores temperatura:", temperature.shape)
print("Dimensión valores datos:", raw_data.shape)


# --------------------------------
# Plot de datos
# --------------------------------

figure = plt.figure(figsize=(8, 8))
axes_temp = figure.add_subplot(221)
axes_temp.plot(range(len(temperature)), temperature, color="red")
axes_temp.set(ylabel="degC", xlabel="time", title="Temperatura")
axes_temp = figure.add_subplot(222)
axes_temp.plot(range(len(raw_data[:, 0:1])), raw_data[:, 0:1], color="blue")
axes_temp.set(ylabel="mbar", xlabel="data", title="Presión")
axes_temp = figure.add_subplot(223)
axes_temp.plot(range(len(raw_data[:, 10:11])), raw_data[:, 10:11], color="green")
axes_temp.set(ylabel="g/m**3", xlabel="data", title="Densidad aire")
axes_temp = figure.add_subplot(224)
axes_temp.plot(range(len(raw_data[:, 11:12])), raw_data[:, 11:12], color="brown")
axes_temp.set(ylabel="m/s", xlabel="data", title="Velocidad viento")
figure.tight_layout()
figure.savefig("JenaClimate.png")

# --------------------------------
# División dataset y normalización
# --------------------------------

num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)

# print(raw_data)
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std
# print(raw_data)

# --------------------------------
# Construcción de los datasets
# --------------------------------

# Prueba de dataset sencillo

# int_sequence = np.arange(10)  # Our raw data [0 1 2 3 4 5 6 7 8 9]
# dummy_dataset = keras.utils.timeseries_dataset_from_array(
#     data=int_sequence[:-3],   # it will be [0 1 2 3 4 5 6] excluding the last three
#     targets=int_sequence[3:], # Target for the sequence that starts at data[N] will be data[N+3]
#     sequence_length=3,        # The sequences will be 3 steps long: [0 1 2], [1 2 3], [2 3 4], ...
#     batch_size=2,             # The sequences will be batched in batches of size 2
# )

# for inputs, targets in dummy_dataset:
#     for i in range(inputs.shape[0]):
#         print([int(x) for x in inputs[i]], int(targets[i]))

# Construcción de train, test y val datasets

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples,
)

# test_dataset = keras.utils.timeseries_dataset_from_array(
#     raw_data[:-delay],
#     targets=temperature[delay:],
#     sampling_rate=sampling_rate,
#     sequence_length=sequence_length,
#     shuffle=True,
#     batch_size=batch_size,
#     start_index=num_train_samples + num_val_samples)

# Comprobamos la forma de los batches
for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break

# --------------------------------
# Naive evaluation
# --------------------------------


def evaluate_naive_method(dataset):
    total_abs_err = 0.0
    samples_seen = 0
    for samples, targets in dataset:
        # samples[:, -1, 1] is the last temperature measurement in the input sequence
        # Data is un-normalized by multiplying it by the sd and adding back the mean
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen


print(f"Train dataset MAE: {evaluate_naive_method(train_dataset):.2f}")
print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
