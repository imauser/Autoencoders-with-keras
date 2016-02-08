import sys
import numpy as np

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



train_error = np.array(train_error)
validation_error = np.array(validation_error)

import matplotlib.pyplot as plt

tem = np.mean(train_error, axis=0)
vem = np.mean(validation_error, axis=0)
tev = np.std(train_error, axis=0)
vev = np.std(validation_error, axis=0)
print(tem.shape)
print(tev.shape)

output_name = sys.argv[1] + ".png"

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.set_xlabel("Epochs")
ax.set_ylabel("Error")
plt.yscale('log')

ax.plot(tem, label="Training error")
ax.fill_between(range(len(tem)), tem+tev, tem-tev, color='gray', alpha=0.5)
#ax.plot(ve, label="Validation error")
lgd = ax.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
fig.savefig(output_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
