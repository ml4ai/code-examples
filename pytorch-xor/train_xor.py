import argparse
#import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import numpy
import torch
import datetime
import os
from pathlib import Path
#import typing


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_timestamp(verbose: bool = False, short: bool = False) -> str:
    """
    Utility to generate timestamps. This is a bit hacky.
    :param verbose: True generates timestamp with component annotations,
                    otherwise concatenated numbers
    :param short: True only includes hours, minutes and seconds
    :return: string representation of the timestamp
    """
    now = datetime.datetime.now()
    if verbose:
        if short:
            return f'{now.hour:02d}:{now.minute:02d}:{now.second:02d}'
        else:
            return f'y={now.year:04d},m={now.month:02d},d={now.day:02d}' \
                   f'_h={now.hour:02d},m={now.minute:02d},s={now.second:02d},' \
                   f'mu={now.microsecond:06d}'
    else:
        if short:
            return f'{now.hour:02d}{now.minute:02d}{now.second:02d}'
        else:
            return f'{now.year:04d}{now.month:02d}{now.day:02d}' \
                   f'_{now.hour:02d}{now.minute:02d}{now.second:02d}' \
                   f'{now.microsecond:06d}'


def write_to_log(log_file: str, msg: str) -> None:
    """
    Utility write a message string to time-stamped line in the log_file.
    :param log_file: Path to log file
    :param msg: string message
    :return: None
    """
    with open(log_file, 'a') as logf:
        logf.write(f'[{get_timestamp(verbose=True,short=True)}] {msg}\n')


# -----------------------------------------------------------------------------
# xor Helpers
# -----------------------------------------------------------------------------

def load_training_data(data_file: str) -> list:
    """
    Helper to load training data from file.
    Assumes data file formatted as csv with rows as follows:
        <target>, <feature-1>, <feature-2>, ...
    Creates data representation prepared for data_to_tensor_pair, as list of:
        ([<feature-1>, <feature-2>, ... ], [<target>])
    :param data_file: Filepath to data file
    :return: Data
    """
    # NOTE: torch expects float data;
    #       default numpy.loadtxt reads as float64,
    #       so specify dtype=numpy.single
    raw = numpy.loadtxt(data_file, dtype=numpy.single, delimiter=',')
    data = list()
    for i in range(raw.shape[0]):
        data.append((raw[i][1:].tolist(), [raw[i][0]]))
    return data


def save_results(output_root: str, loss_values: list, accuracy_values: list) -> None:
    """
    Helper to save model training results to file.
    Results file format per row:
        <loss_value>, <accuracy_value>
    :param output_root: Output root path
    :param loss_values: list of model training loss_values
    :param accuracy_values: list of model training accuracy_values
    :return: None
    """
    results_file = os.path.join(output_root, 'results.csv')
    results = numpy.array([loss_values, accuracy_values]).transpose()
    numpy.savetxt(results_file, results, delimiter=',')


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


def train_xor(data, learning_rate, iterations, log_file):
    """

    :param data:
    :param learning_rate: Model learning rate
    :param iterations: Number of training iteratiosn
    :param log_file: Log file path
    :return:
    """

    if torch.cuda.is_available():
        write_to_log(log_file, 'CUDA is available -- using GPU')
        device = torch.device('cuda')
    else:
        write_to_log(log_file, 'CUDA is NOT available -- using CPU')
        device = torch.device('cpu')

    # Define our toy training data set for the XOR function.
    training_data = data_to_tensor_pair(data, device)

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

'''
def plot(loss_values: list, accuracy_values: list,
         save_file: typing.Optional[str] = None,
         show_p: bool = False) -> None:
    """
    Helper to plot the the loss and accuracy.
    :param loss_values: Sequence of training loss values
    :param accuracy_values: Sequence of model training accuracy values
    :param save_file: Filepath to save image to (does not save if None, the default)
    :param show_p: Flag for whether to show the image after rendering (default False)
    :return: None
    """
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
'''


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    """
    The top-level function to manage the experiment.
    Processes the script arguments
    Creates results root path (but throws error if already exists).
    Writes to log message before and after each experiment step.
    :return: None
    """

    # process script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=100,
                        help='number of training iterations.')
    parser.add_argument('--learning-rate', type=float, default=1.0,
                        help='learning rate.')
    parser.add_argument('--data-file', default='xor.txt',
                        help='path to data file.')
    parser.add_argument('--output-root', default='xor_results',
                        help='path to root of output results directory.')
    parser.add_argument('--gen-output-timestamp', action='store_true',
                        help='flag for whether to generate the output results '
                             'directory with a unique timestamp.')
    parser.add_argument('--save-model', action='store_true',
                        help='flag for whether to save the generated model.')
    args = parser.parse_args()

    if args.gen_output_timestamp:
        output_root = f'{args.output_root}_{get_timestamp()}'
    else:
        output_root = args.output_root

    # create output_root directory path but throw error if it already exists
    Path(output_root).mkdir(parents=True, exist_ok=False)

    log_file = os.path.join(output_root, 'log.txt')

    write_to_log(log_file, f'[{get_timestamp(verbose=True)}]')
    write_to_log(log_file, 'XOR training script START')

    write_to_log(log_file, 'Loading training data')
    data = load_training_data(args.data_file)
    write_to_log(log_file, 'Loading training data DONE')

    write_to_log(log_file, 'Train XOR')
    model, loss_values, accuracy_values = \
        train_xor(data, args.learning_rate, args.iterations, log_file)
    write_to_log(log_file, 'Train XOR DONE')

    write_to_log(log_file, 'Saving results')
    save_results(output_root, loss_values, accuracy_values)
    write_to_log(log_file, 'Saving results DONE')

    if args.save_model:
        model_dst_file = os.path.join(output_root, 'xor_model.pt')
        write_to_log(log_file, f'Saving model to {model_dst_file}')
        torch.save(model.state_dict(), model_dst_file)
        write_to_log(log_file, 'Saving model DONE')

    # Plotting should be done as post-processing after HPC run
    # plot_file = os.path.join(output_root, 'train_xor_plot.png')
    # plot(loss_values, accuracy_values, save_file=plot_file)

    write_to_log(log_file, 'XOR training script DONE.')
    write_to_log(log_file, f'[{get_timestamp(verbose=True)}]')


# -----------------------------------------------------------------------------
# Script
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    main()
