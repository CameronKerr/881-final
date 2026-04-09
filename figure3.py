############
# Figure 3 #
############

# Code to generate Figure 3a and b

# Required packages
import numpy as np
import matplotlib.pyplot as plt

# ------ Benefit graph ------ #
# Set parameters
Sp = 1
alpha = 1
h = 0.5
omega = 1
k = 0.3
D=1

cjs = np.linspace(0,1,75)
epsilon = 1e-8

for s in [0,5,10]:
    dbs = []
    for cj in cjs:
        db = (b(cj + epsilon, s, h, Sp, alpha) - b(cj, s, h, Sp, alpha))/epsilon
        db = db*Sp/(alpha + D*4)
        dbs.append(db)
    plt.plot(cjs, dbs, label="s = " + str(s), lw=3)

plt.hlines(k, xmin=0, xmax=1, linestyle='dotted', lw=3, label='k')
plt.legend()
plt.xlabel('Mutant node concentration')
plt.ylabel('Benefit to production')
plt.grid(True)
plt.savefig("derivativegraph.pdf", dpi=300)
plt.show()

## ------ Half-saturation point plot ------ ##
# Set parameters
Sp = 1
alpha = 1
omega = 1
k = 0.3
D=1

## -- Plot over h -- ##
hs = np.linspace(0,1,75)
G = nx.complete_graph(5)
for s in [0,5,10]:
    pfixs = []
    iteration = 0
    for h in hs:
        prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)
        pfixs.append(prob)
        iteration += 1
        print(iteration) 
        
    plt.plot(hs, pfixs,label='s = ' + str(s), lw=3)
plt.xlabel('Half-saturation point (h)')
plt.ylabel('Fixation probability')
plt.legend()
plt.grid(True)
plt.savefig("saturationpoint.pdf", dpi=300)
plt.show()

