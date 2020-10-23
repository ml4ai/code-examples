import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy
import torch
import datetime
import os
from pathlib import Path


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_timestamp(verbose=False):
    now = datetime.datetime.now()
    if verbose:
        return f'y={now.year:04d},m={now.month:02d},d={now.day:02d}' \
               f'_h={now.hour:02d},m={now.minute:02d},s={now.second:02d},mu={now.microsecond:06d}'
    else:
        return f'{now.year:04d}{now.month:02d}{now.day:02d}' \
               f'_{now.hour:02d}{now.minute:02d}{now.second:02d}{now.microsecond:06d}'


def write_to_log(log_file, msg):
    with open(log_file, 'a') as logf:
        logf.write(f'{msg}\n')


# -----------------------------------------------------------------------------
# xor Helpers
# -----------------------------------------------------------------------------

def load_training_data(data_file):
    # NOTE: torch expects float data;
    #       default numpy.loadtxt reads as float64,
    #       so specify dtype=numpy.single
    raw = numpy.loadtxt(data_file, dtype=numpy.single, delimiter=',')
    data = list()
    for i in range(raw.shape[0]):
        data.append((raw[i][1:].tolist(), [raw[i][0]]))
    return data


def data_to_tensor_pair(data, device):
    x = torch.tensor([x for x, y in data], device=device)
    y = torch.tensor([y for x, y in data], device=device)
    return x, y


def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        return compute_accuracy(y_pred, y)


def compute_accuracy(predictions, expected):
    correct = 0
    total = 0
    for y_pred, y in zip(predictions, expected):
        correct += round(y_pred.item()) == round(y.item())
        total += 1
    return correct / total


def construct_model(hidden_units, num_layers):
    layers = []
    prev_layer_size = 2
    for layer_no in range(num_layers):
        layers.extend([
            torch.nn.Linear(prev_layer_size, hidden_units),
            torch.nn.Tanh()
        ])
        prev_layer_size = hidden_units
    layers.extend([
        torch.nn.Linear(prev_layer_size, 1),
        torch.nn.Sigmoid()
    ])
    return torch.nn.Sequential(*layers)


def train_xor(data_path, learning_rate, iterations, log_file):

    if torch.cuda.is_available():
        write_to_log(log_file, 'CUDA is available -- using GPU')
        device = torch.device('cuda')
    else:
        write_to_log(log_file, 'CUDA is NOT available -- using CPU')
        device = torch.device('cpu')

    # Define our toy training data set for the XOR function.
    training_data = data_to_tensor_pair(load_training_data(data_path), device)

    # Define our model. Use default initialization.
    model = construct_model(hidden_units=10, num_layers=2)
    model.to(device)

    loss_values = []
    accuracy_values = []
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    for iter_no in range(iterations):
        write_to_log(log_file, f'iteration #{iter_no + 1}')
        # Perform a parameter update.
        model.train()
        optimizer.zero_grad()
        x, y = training_data
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss_value = loss.item()
        write_to_log(log_file, f'  loss: {loss_value}')
        loss_values.append(loss_value)
        loss.backward()
        optimizer.step()
        # Evaluate the model.
        accuracy = evaluate_model(model, x, y)
        write_to_log(log_file, f'  accuracy: {accuracy:.2%}')
        accuracy_values.append(accuracy)

    return model, loss_values, accuracy_values


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

def plot(loss_values, accuracy_values, save_file=None, show_p=False):
    # Plot loss and accuracy.
    fig, ax = plt.subplots()
    ax.set_title('Loss and Accuracy vs. Iterations')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.set_xlim(left=1, right=len(loss_values))
    ax.set_ylim(bottom=0.0, auto=None)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_array = numpy.arange(1, len(loss_values) + 1)
    loss_y_array = numpy.array(loss_values)
    left_plot = ax.plot(x_array, loss_y_array, '-', label='Loss')
    right_ax = ax.twinx()
    right_ax.set_ylabel('Accuracy')
    right_ax.set_ylim(bottom=0.0, top=1.0)
    accuracy_y_array = numpy.array(accuracy_values)
    right_plot = right_ax.plot(x_array, accuracy_y_array, '--', label='Accuracy')
    lines = left_plot + right_plot
    ax.legend(lines, [line.get_label() for line in lines])

    if save_file is not None:
        plt.savefig(save_file)

    if show_p:
        plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=1.0)
    parser.add_argument('--data-file', default='xor.txt')
    parser.add_argument('--output-root', default='')
    parser.add_argument('--gen-output-timestamp', default=True)
    parser.add_argument('--save-model', default=True)
    args = parser.parse_args()

    if args.gen_output_timestamp:
        output_root = os.path.join(args.output_root, f'xor_results_{get_timestamp()}')
    else:
        output_root = args.output_root

    # create output_root directory path if does not already exist
    Path(output_root).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(output_root, 'log.txt')
    plot_file = os.path.join(output_root, 'train_xor_plot.png')

    write_to_log(log_file, 'XOR training script START')

    model, loss_values, accuracy_values = \
        train_xor(args.data_file, args.learning_rate, args.iterations, log_file)

    if args.save_model is not None:
        model_dst_file = os.path.join(output_root, 'xor_model.pt')
        write_to_log(log_file, f'Saving model to {model_dst_file}')
        torch.save(model.state_dict(), model_dst_file)

    plot(loss_values, accuracy_values, save_file=plot_file)

    write_to_log(log_file, 'XOR training script DONE.')


# -----------------------------------------------------------------------------
# Script
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
