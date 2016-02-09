#!/usr/bin/python3
import sys
import os
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename) as f:
        lines = f.readlines()

    digits = 6

    train_error = []
    validation_error = []

    for line in lines:
        if " loss:" in line:
            pos = line.rfind(" loss:")
            offset = len(" loss:") + 1
            realdigits = digits
            if line[pos+offset+digits] == "e":
                realdigits += 4
            train_error.append(float(line[pos+offset:pos+offset+realdigits]))
        if " val_loss:" in line:
            pos = line.find(" val_loss:")
            offset = len(" val_loss:") + 1
            validation_error.append(float(line[pos+offset:pos+offset+digits]))

    return train_error, validation_error


def plot(errors, output_name):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Error")
    plt.yscale('log')
    for key in errors:
        ax.plot(errors[key][0], label="Training error" + key)
        ax.plot(errors[key][1], label="Validation error" + key)
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.savefig(output_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
    dirname = sys.argv[1]
    os.chdir(dirname)
    errors = dict()
    for f in os.listdir():
            if f.endswith(".txt"):
                errors[f] = read_file(f)
    plot(errors, "output" + ".png")
