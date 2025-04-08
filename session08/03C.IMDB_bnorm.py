
from keras.preprocessing import sequence
from keras.datasets import imdb

import torch
import numpy as np

import sys

print()
print("------------------------------------")

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("GPU available:", torch.cuda.get_device_name(0))
else:
    print("ERROR: no GPU available")
    sys.exit(0)
    #device = torch.device('cpu')

num_words = 20000 # vocabulary size
maxlen = 80  # max length of reviews
batch_size = 512
num_epochs = 20
random_index = np.random.randint(0,5000)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print('Loading data...')
(x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = imdb.load_data(num_words=num_words)
print(len(x_train_orig), 'train sequences')
print(len(x_test_orig), 'test sequences')
print('Original train set shape:', x_train_orig.shape)
print('Original test set shape:', x_test_orig.shape)
print("Tokenized review: ", x_train_orig[random_index])

print('Pad sequences (samples x time)')
x_train_padded = sequence.pad_sequences(x_train_orig, maxlen=maxlen)
print('Padded train set shape:', x_train_padded.shape)
print("Padded review: ", x_train_padded[random_index])

x_train = x_train_padded[:15000]
y_train = y_train_orig[:15000]
x_val = x_train_padded[15000:]
y_val = y_train_orig[15000:]


print('No. of train samples:', len(x_train))
print('No. of validation samples:', len(x_val))




# --------------------------------------------------
# Reverse a sentence
# --------------------------------------------------

# word_index = imdb.get_word_index()
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train_orig[random_index]])
# print("Decoded review: ",decoded_review)
# if (y_train_orig[random_index]==0):
#     print("Negative review")
# else:
#     print("Positive review")


# --------------------------------------------------
# Vectorize training and testing data
# Change labels data types
# --------------------------------------------------

def vectorize_sequences(sequences, dimension = num_words):
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train2 = vectorize_sequences(x_train)
x_val2 = vectorize_sequences(x_val)
y_train2 = np.asarray(y_train).astype('float32')
y_val2 = np.asarray(y_val).astype('float32')



# --------------------------------------------------
# Builds the neural network
# --------------------------------------------------

net = torch.nn.Sequential(
      torch.nn.Linear(num_words, 16),
      torch.nn.ReLU(),
      torch.nn.BatchNorm1d(16),
      torch.nn.Linear(16, 16),
      torch.nn.ReLU(),
      torch.nn.BatchNorm1d(16),
      torch.nn.Linear(16, 1),
      torch.nn.Sigmoid()
      ).to(device)

optimizer = torch.optim.RMSprop(net.parameters(), lr = 0.001)
criterion = torch.nn.BCELoss()


# --------------------------------------------------
# Main loop
# --------------------------------------------------

loss_v = np.zeros(num_epochs)
loss_val_v = np.zeros(num_epochs)
accuracy_v = np.zeros(num_epochs)
accuracy_val_v = np.zeros(num_epochs)

num_batches = len(x_train2) // batch_size

input_val = torch.tensor(x_val2, dtype=torch.float32).to(device)
labels_val = torch.tensor(y_val2, dtype=torch.float32).to(device)

for epoch in range(num_epochs):
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    net.train()
    for i in range(num_batches):
        
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        inputs = torch.tensor(x_train2[batch_start:batch_end], dtype=torch.float32).to(device)
        labels = torch.tensor(y_train2[batch_start:batch_end], dtype=torch.float32).to(device)

        # forward + backward + optimize
        outputs = net(inputs).flatten()
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        total_loss += loss.item()
        mask1 = (outputs>0.5) & (labels>0.5)
        correct_predictions += sum(mask1)
        mask2 = (outputs<=0.5) & (labels<=0.5)
        correct_predictions += sum(mask2)
        total_samples += len(inputs)

    
    net.eval()
    with torch.no_grad():
        outputs_val = net(input_val).flatten()
        loss_val_v[epoch] = criterion(outputs_val, labels_val).item()

    correct_predictions_val = 0
    mask_val1 = (outputs_val>0.5) & (labels_val>0.5)
    correct_predictions_val += sum(mask_val1)
    mask_val2 = (outputs_val<=0.5) & (labels_val<=0.5)
    correct_predictions_val += sum(mask_val2)
    accuracy_val_v[epoch] = correct_predictions_val / len(input_val)

        
    loss_v[epoch] = total_loss / num_batches
    accuracy_v[epoch] = correct_predictions / total_samples
    
    print("Epoch {:02d}: loss {:.4e} - accuracy {:.4f} - val. loss {:.4e} - val. accuracy {:.4f}".format(epoch+1, loss_v[epoch], 100*accuracy_v[epoch], loss_val_v[epoch], 100*accuracy_val_v[epoch]))


import matplotlib.pyplot as plt

epochs = range(1, num_epochs + 1)

plt.figure()
plt.plot(epochs, loss_v, 'b-o', label='Training ')
plt.plot(epochs, loss_val_v, 'r-o', label='Validation ') 
plt.title('Training and validation loss (batch norm.)')
plt.xlabel('Epochs')
plt.legend()
plt.savefig("03C.IMDB_bnorm.Loss.png")

plt.figure()
plt.plot(epochs, accuracy_v, 'b-o', label='Training ')
plt.plot(epochs, accuracy_val_v, 'r-o', label='Validation ') 
plt.title('Training and validation accuracy (batch norm.)')
plt.xlabel('Epochs')
plt.legend()
plt.savefig("03C.IMDB_bnorm.Accuracy.png")