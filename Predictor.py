# This file contains the bulk of the interesting parts of this project.

from Lib import *
from sklearn.model_selection import KFold, LeaveOneOut
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import random
import os

def read_dataset(dataset):
    # Read in files
    positives = read_file_in_dataset(dataset, "positive")
    negatives = read_file_in_dataset(dataset, "negative")
    #negatives = [n for n in negatives if n not in positives]
    #negatives = negatives[:1000]

    # Construct full lists
    x_data = positives + negatives
    y_data = ([[1]] * len(positives)) + ([[0]] * len(negatives))

    return x_data, y_data

def construct_datasets(training_set, max_len=None):
    train_x, train_y = read_dataset(training_set)

    paired = list(zip(train_x, train_y))
    if max_len != None:
        paired = list(filter(lambda x: len(x[0]) <= max_len, paired))
    #random.shuffle(paired)
    train_x, train_y = zip(*paired)

    # Convert values to numbers
    train_x = [[VOCAB[c][0] for c in p] for p in train_x]

    return train_x, np.array(train_y)

def MCC(FN, FP, TN, TP):
    mccs = []
    for (fn, fp, tn, tp) in zip(FN, FP, TN, TP):
        mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mccs.append(mcc)
    return mccs

plot_num = 0
def plot(epochs, measure, val_measure, color, name):
    global plot_num
    plt.figure(plot_num)
    plt.plot(epochs, measure, color + 'o', label='Training {}'.format(name))
    plt.plot(epochs, val_measure, color, label='Validation {}'.format(name))
    plt.title('Training and Validation {}'.format(name))
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.legend()
    plt.show(block=False)
    plot_num += 1

BATCHES=3
def create_model(transfer_X, transfer_Y, seq_len, seed):
    # CNN branch
    input = tf.keras.Input(shape=(seq_len))
    nw1 = layers.Embedding(len(VOCAB), 55, embeddings_regularizer=tf.keras.regularizers.l1(0.01))(input)
    nw1 = layers.Dropout(0.3)(nw1)
    #nw1 = layers.BatchNormalization()(nw1)
    nw1 = layers.Conv1D(83, 19)(nw1)
    nw1 = layers.Dropout(0.4)(nw1)
    #nw1 = layers.BatchNormalization()(nw1)
    nw1 = layers.Bidirectional(layers.LSTM(125))(nw1)

    # RNN branch
    nw2 = layers.Embedding(len(VOCAB), 144, embeddings_regularizer=tf.keras.regularizers.l2(0.0001))(input)
    nw2 = layers.Dropout(0.45)(nw2)
    #nw2 = layers.BatchNormalization()(nw2)
    nw2 = layers.Bidirectional(layers.LSTM(106))(nw2)

    # Final merging
    nw = layers.Concatenate()([nw1, nw2])
    nw = layers.Dropout(0.1)(nw)
    #nw = layers.BatchNormalization()(nw)
    nw = layers.Dense(100, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1(0.0001))(nw)
    nw = layers.Dropout(0.05)(nw)
    #nw = layers.BatchNormalization()(nw)
    nw = layers.Dense(1, activation='sigmoid')(nw)

    model = tf.keras.Model(inputs=input, outputs=nw)
    model.compile(optimizer='adam', loss='binary_crossentropy',
        metrics = ['acc', 'FalseNegatives', 'FalsePositives', 'TrueNegatives', 'TruePositives'])
    model.summary()

    if not os.path.exists("models/{}.h5".format(seed)):
        transfer_set = tf.data.Dataset.from_tensor_slices((transfer_X, transfer_Y)).batch(BATCHES, drop_remainder = True)
        transfer_set = transfer_set.shuffle(len(transfer_X))
        model.fit(transfer_set, epochs=95, verbose=1)
        model.save_weights("models/{}.h5".format(seed))
    else:
        model.load_weights("models/{}.h5".format(seed))

    return model

# Trains and tests the model
def test_model(train, test):
    # Create datasets
    train_X, train_Y = construct_datasets(train)
    transfer_X, transfer_Y = construct_datasets("dataset_afp")
    test_X, test_Y = construct_datasets(test)

    # Pad them
    max_len = max(map(len, train_X + transfer_X + test_X))
    [train_X, transfer_X, test_X] = [tf.keras.preprocessing.sequence.pad_sequences(x, value = VOCAB[' '][0], padding = 'post', maxlen=max_len) for x in [train_X, transfer_X, test_X]]

    for i in range(30):
        np.random.seed(i)
        tf.random.set_seed(i)

        # Create the model
        model = create_model(transfer_X, transfer_Y, max_len)

        # Batch the data
        train_set = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(BATCHES, drop_remainder = True)
        train_set = train_set.shuffle(len(train_X))
        test_set = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(BATCHES, drop_remainder = True)
        test_set = test_set.shuffle(len(test_X))

        # Train
        _ = model.fit(train_set, validation_data=test_set, epochs=90, verbose=1)

        # Evaluate
        results = model.evaluate(test_set)
        [_, accuracy, FN, FP, TN, TP] = results
        mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        recall = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        f1_score = 2 * precision * recall / (precision + recall)

        with open("results.tab", "a") as file:
            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(test, i, mcc, accuracy, recall, specificity, precision, f1_score))

# Only performs 10-fold cross validation on the provided set.
def kfold(train):
    # Create datasets
    X, Y = construct_datasets(train)
    transfer_X, transfer_Y = construct_datasets("dataset_afp")

    # Pad them
    max_len = max(map(len, X + transfer_X))
    [X, transfer_X] = [tf.keras.preprocessing.sequence.pad_sequences(x, value = VOCAB[' '][0], padding = 'post', maxlen=max_len) for x in [X, transfer_X]]

    for i in range(30):
        np.random.seed(i)
        tf.random.set_seed(i)

        # Create model
        model = create_model(transfer_X, transfer_Y, max_len, i)

        counter = 0
        accs = []
        losses = []
        mccs = []
        val_accs = []
        val_losses = []
        val_mccs = []
        total_mcc = total_acc = total_recall = total_spec = total_prec = total_f1 = 0.0
        for train_index, test_index in KFold(NUM_FOLDS, shuffle=True, random_state=i).split(X):
            train_x, test_x = X[train_index], X[test_index]
            train_y, test_y = Y[train_index], Y[test_index]

            train_set = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(BATCHES, drop_remainder = True)
            train_set = train_set.shuffle(len(train_x))
            test_set = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(BATCHES, drop_remainder = True)
            test_set = test_set.shuffle(len(test_x))

            # Train
            model.load_weights("models/{}.h5".format(i))
            history = model.fit(train_set, validation_data=test_set, epochs=90, verbose=1)
            history_dict = history.history

            accs.append(history_dict['acc'])
            losses.append(history_dict['loss'])
            mccs.append(MCC(
                history_dict["FalseNegatives"],
                history_dict["FalsePositives"],
                history_dict["TrueNegatives"],
                history_dict["TruePositives"]))

            val_accs.append(history_dict['val_acc'])
            val_losses.append(history_dict['val_loss'])
            val_mccs.append(MCC(
                history_dict["val_FalseNegatives"],
                history_dict["val_FalsePositives"],
                history_dict["val_TrueNegatives"],
                history_dict["val_TruePositives"]))

            # Evaluate
            results = model.evaluate(test_set)
            [_, accuracy, FN, FP, TN, TP] = results
            mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            recall = TP / (TP + FN)
            specificity = TN / (TN + FP)
            precision = TP / (TP + FP)
            f1_score = 2 * precision * recall / (precision + recall)

            with open("results/f{}_{}.txt".format(i, counter), "w") as file:
                file.write("MCC: {}\n".format(mcc))
                file.write("Accuracy: {}\n".format(accuracy))
                file.write("Recall/Sensitivity: {}\n".format(recall))
                file.write("Specificity: {}\n".format(specificity))
                file.write("Precision: {}\n".format(precision))
                file.write("F1 Score: {}\n".format(f1_score))

            total_mcc += mcc
            total_acc += accuracy
            total_recall += recall
            total_spec += specificity
            total_prec += precision
            total_f1 += f1_score

            counter += 1

        with open("results.tab", "a") as file:
            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                train, 
                i, 
                total_mcc / NUM_FOLDS,
                total_acc / NUM_FOLDS,
                total_recall / NUM_FOLDS,
                total_spec / NUM_FOLDS,
                total_prec / NUM_FOLDS,
                total_f1 / NUM_FOLDS))

        with open("results/t{}.txt".format(i), "w") as file:
            file.write("Average MCC: {}\n".format(total_mcc / NUM_FOLDS))
            file.write("Average Accuracy: {}\n".format(total_acc / NUM_FOLDS))
            file.write("Average Recall/Sensitivity: {}\n".format(total_recall / NUM_FOLDS))
            file.write("Average Specificity: {}\n".format(total_spec / NUM_FOLDS))
            file.write("Average Precision: {}\n".format(total_prec / NUM_FOLDS))
            file.write("Average F1 Score: {}\n".format(total_f1 / NUM_FOLDS))

        '''
        accs = [sum(i) / NUM_FOLDS for i in zip(*accs)]
        losses = [sum(i) / NUM_FOLDS for i in zip(*losses)]
        mccs = [sum(i) / NUM_FOLDS for i in zip(*mccs)]
        val_accs = [sum(i) / NUM_FOLDS for i in zip(*val_accs)]
        val_losses = [sum(i) / NUM_FOLDS for i in zip(*val_losses)]
        val_mccs = [sum(i) / NUM_FOLDS for i in zip(*val_mccs)]

        epochs = range(1, len(accs) + 1)
        plot(epochs, mccs, val_mccs, 'g', "MCC")
        plot(epochs, accs, val_accs, 'b', "Accuracy")
        plot(epochs, losses, val_losses, 'r', "Loss")
        plt.show()
        '''

def main():
    if len(sys.argv) == 2:
        train = sys.argv[1]
        kfold(train)
    elif len(sys.argv) == 3:
        train = sys.argv[1]
        test = sys.argv[2]
        test_model(train, test)
    else:
        print("Usage: python {} train_set [test_set]".format(sys.argv[0]))

if __name__ == "__main__":
    main()
