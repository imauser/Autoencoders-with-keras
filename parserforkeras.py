import sys

with open(sys.argv[1]) as f:
    lines = f.readlines()

digits = 6

train_error = []
validation_error = []

for line in lines:
    if " loss:" in line:
        pos = line.find(" loss:")
        offset = len(" loss:") + 1
        realdigits = digits
        if line[pos+offset+digits] == "e":
            realdigits += 4
        train_error.append(float(line[pos+offset:pos+offset+realdigits]))
    if " val_loss:" in line:
        pos = line.find(" val_loss:")
        offset = len(" val_loss:") + 1
        validation_error.append(float(line[pos+offset:pos+offset+digits]))



import matplotlib.pyplot as plt


te = train_error
ve = validation_error

output_name = sys.argv[1] + ".png"

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlabel("Epochs")
ax.set_ylabel("Error")
plt.yscale('log')
ax.plot(te, label="Training error")
ax.plot(ve, label="Validation error")
lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
fig.savefig(output_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
