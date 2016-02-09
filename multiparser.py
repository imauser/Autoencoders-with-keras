import sys
import numpy as np

import matplotlib.pyplot as plt

train_error = []
validation_error = []

digits = 5

for i in range(len(sys.argv)-1):
    with open(sys.argv[i+1]) as f:
        lines = f.readlines()
    train_error.append([])
    validation_error.append([])

    for line in lines:
        if " loss:" in line:
            pos = line.find(" loss:")
            offset = len(" loss:") + 1
            train_error[i].append(float(line[pos+offset:pos+offset+digits]))
        if " val_loss:" in line:
            pos = line.find(" val_loss:")
            offset = len(" val_loss:") + 1
            validation_error[i].append(float(line[pos+offset:pos+offset+digits]))

te = np.array(train_error)
ve = np.array(validation_error)

tem = np.mean(te, axis=0)
vem = np.mean(ve, axis=0)
tev = np.std(te, axis=0)
vev = np.std(ve, axis=0)

output_name = sys.argv[1] + ".png"

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlabel("Epochs")
ax.set_ylabel("Error")
#plt.yscale('log')

#ax.plot(tem, label="Training error")
ax.plot(vem, label="Validation error")
ax.fill_between(range(len(vem)), vem - vev, vem + vev, facecolor='blue', alpha=0.5)
lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
fig.savefig(output_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
