#!/usr/bin/python2.7
import sys
import os
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename) as f:
        lines = f.read().split("\n")

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

    print("errors for " + filename)
    print("lines: " + str(len(lines)))
    print("trainlen: " + str(len(train_error)))
    print("vallen: " + str(len(validation_error)))
    return train_error, validation_error


def plot(errors, output_name):

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Error")
    plt.yscale('log')
    for key in errors:
        ax.plot(errors[key][0], label="Training error" + key[0:len(key)-4])
        ax.plot(errors[key][1], label="Validation error" + key[0:len(key)-4])
    lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    fig.savefig(output_name, bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
    dirname = sys.argv[1]
    os.chdir(dirname)
    errors = dict()
    for file in os.listdir("."):
            if file.endswith(".txt"):
                errors[file] = read_file(file)
    plot(errors, "output" + ".png")
