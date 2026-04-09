############
# Figure 2 #
############

# Required packages
import networkx as nx
import numpy as np
import random

# Set parameters
Sp = 1
alpha = 1
s = 0
h = 0.5
omega = 0.5

##------- 2D HEATMAP ----------##

# Generating D and k heatmap
ks = np.linspace(0,1,75)
Ds = np.linspace(0,1,75)

# Function to generate star graph
#G = nx.star_graph(4)
G = nx.cycle_graph(5)
K = nx.complete_graph(5)

k_values = []
D_values = []
ratios = []

iteration = 0
for k in ks:
    for D in Ds:
        phi_neutral = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega=0)
        prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)
        k_values.append(k)
        D_values.append(D)
        ratios.append(prob/phi_neutral)
        iteration += 1
        print(iteration)

avg_deg = sum(d for n, d in G.degree()) / G.number_of_nodes()

plt.scatter(k_values, D_values, c=ratios, cmap = 'RdBu', vmin=0.5, vmax=1.5)
k_curve = alpha / (alpha + Ds*avg_deg)
plt.xlim(0,1)
plt.ylim(0,1)
plt.plot(k_curve, Ds, 'k--', label='Linear theory')
cbar = plt.colorbar(label='Fixation probability ratio')
plt.legend()
plt.xlabel('Cost of Production (k)')
plt.ylabel('Diffusion Coefficient (D)')
plt.savefig("regulargraph_s10.pdf", dpi=300)
plt.show()

##---------- Star graph curve ------------##

# Requires change in pfix_diffusion to alternative output for each node

# Generating D and k heatmap
ks = np.linspace(0,1,75)
# Function to generate star graph
G = nx.star_graph(4)
K = nx.complete_graph(5)

pfix_center_d5 = []
pfix_leaf_d5 = []

D=1
iteration = 0
for k in ks:
    prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)
    phi_neutral = pfix_diffusion(G, D, Sp, alpha, s, h, k, 0)
    pfix_center_d5.append(prob[0]/phi_neutral[0])
    pfix_leaf_d5.append(prob[1]/phi_neutral[1])
    iteration += 1
    print(iteration) 
    
plt.plot(ks, pfix_center_d5, 'b', label='Center node', lw=3)
plt.plot(ks, pfix_leaf_d5, 'r', label='Leaf node', lw=3)
plt.xlabel('Cost of production (k)')
plt.ylabel('Fixation probability ratio')
plt.legend()
plt.grid(True)
plt.savefig("stargraph.pdf", dpi=300)
plt.show()
