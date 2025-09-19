import os
import matplotlib.pyplot as plt
import numpy as np
benchmarks = ['cartpole', 'cartpole_stab', 'cartdoublepole', 'cartdoublepole_stab', 'swimmer']

curves = {}

for b in benchmarks:
    files = [f for f in os.listdir(os.getcwd()) if os.path.isfile(f) and f.startswith(b)]
    for f in files:
        print(f)
        
        curve = np.loadtxt(f, skiprows=1, delimiter=",")[:, -1]
        print(curve)

        if b in curves.keys():
            curves[b].append(curve)
        else:
            curves[b] = [curve]

for b in benchmarks:
    c = curves[b]
    min_len = min([len(c_) for c_ in c])
    for k in range(len(c)):
        curves[b][k] = c[k][:min_len]


plt.figure()
for b in curves.keys():
    m = np.mean(np.array(curves[b]), axis=0)
    std = np.std(np.array(curves[b]), axis=0)
    
    plt.plot(m, label=b)
    plt.fill_between(np.arange(len(m)), m-std, m+std, alpha=0.2)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('DPETS.pdf')
plt.show()