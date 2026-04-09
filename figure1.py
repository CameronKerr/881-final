#############
## Figure 1 #
#############

# Code to reproduce Figure 1

# Required packages
import networkx as nx
import numpy as np
import random

# Functions to generate and calculate fixation probability and amplification factor of graphs
# Modified star graphs
def mod_star(N, m):
    G = nx.star_graph(N-1) # Generate star graph
    not_edges = list(nx.non_edges(G)) # Edges not in G
    new_edges_index = np.random.choice(range(0, len(not_edges)), size=m, replace=False)
    new_edges = [not_edges[i] for i in new_edges_index]
    G.add_edges_from(new_edges) # Add new edges to graph
    
    # Calculate amplification and fixation probability
    amp = amplification_factor(G)
    prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)
    
    return prob, amp
# Erdos Renyi random networks
def er_graph(N, p):
    G = nx.erdos_renyi_graph(N, p) # Generate ER graph
    while not nx.is_connected(G): # Ensure the graph is connected
        G = nx.erdos_renyi_graph(N, p)

    # Calculate amplification and fixation probability
    amp = amplification_factor(G)
    prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)
    
    return prob, amp
# Random geometric graphs
def rg_graph(N, p):
    G = nx.random_geometric_graph(N, p)
    while not nx.is_connected(G): # Ensure graph is connected
        G = nx.random_geometric_graph(N, p)
        
    # Calculate amplification and fixation probability
    amp = amplification_factor(G)
    prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)

    return prob, amp
# Bipartite graph
def bipartite_graph(N, section, p):
    G = nx.algorithms.bipartite.random_graph((N-section), section, p) # Generate bipartite graph with N-k and k length sections
    while not nx.is_connected(G): # Ensure graph is connected
        G = nx.algorithms.bipartite.random_graph(N-section, section, p)
    
    # Calculate amplification and fixation probability
    amp = amplification_factor(G)
    prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)
    
    return prob, amp
# Detour graph
def detour_graph(N, path_length):
    G = nx.complete_graph(N-path_length+1) # Start with complete graph
    random_edge = list(G.edges)[random.randint(0, G.size()-1)]
    G.remove_edge(random_edge[0], random_edge[1]) # Remove random edge
    new_path_nodes = [random_edge[0]] + list(range(len(G), len(G) + path_length - 1)) + [random_edge[1]]
    nx.add_path(G, new_path_nodes) # Add in path where edge used to be
    
    # Calculate amplification and fixation probability
    amp = amplification_factor(G)
    prob = pfix_diffusion(G, D, Sp, alpha, s, h, k, omega)
    
    return prob, amp
    
# Function from Ma et al. to approximate amplification factor
def amplification_factor(G):
    dlist = dict(G.degree())
    N = len(dlist)
    corr = np.zeros((N, N))
    p = np.zeros(N)

    for d in dlist:
        p[dlist[d]] += 1

    for e in G.edges:
        d0 = dlist[e[0]]
        d1 = dlist[e[1]]
        corr[d0, d1] = corr[d0, d1] + 1 / p[d0] / d0
        corr[d1, d0] = corr[d1, d0] + 1 / p[d1] / d1

    p = p / N

    idx = np.nonzero(p)[0]
    p = p[idx]
    corr = corr[idx][:, idx]
    
    
    amp = (p.T @ corr @ (1 / idx)[:,None]) / (p.T @ corr @ (1 / idx**2)[:,None])
    amp = (amp / idx) @ p

    return amp


# Set parameters
Sp = 1
alpha = 1
s = 0.01
h = 0.5
omega = 0.5
D = 1
k=1 # Beneficial mutant

# Helper function to get first or second elements
def divide(l):
    l0 = [x[0] for x in l]
    l1 = [x[1] for x in l]
    return l0, l1

#------- Generate graphs to consider

# Graph parameters
n_iter = 10
N = 10

# Collect probabilities and factors
star_prob, star_amp = divide([mod_star(N, 0) for _ in range(n_iter)] + 
                             [mod_star(N, 1) for _ in range(n_iter)] + 
                             [mod_star(N, 2) for _ in range(n_iter)] + 
                             [mod_star(N, 3) for _ in range(n_iter)])
er_prob, er_amp = divide([er_graph(N, 0.2) for _ in range(n_iter)] + 
                         [er_graph(N, 0.4) for _ in range(n_iter)] + 
                         [er_graph(N, 0.6) for _ in range(n_iter)] + 
                         [er_graph(N, 0.7) for _ in range(n_iter)])
rg_prob, rg_amp = divide([rg_graph(N, 0.2) for _ in range(n_iter)] +
                         [rg_graph(N, 0.4) for _ in range(n_iter)] +
                         [rg_graph(N, 0.6) for _ in range(n_iter)])
bp_prob, bp_amp = divide([bipartite_graph(N, 1, 0.4) for _ in range(n_iter)] + 
                         [bipartite_graph(N, 2, 0.4) for _ in range(n_iter)] + 
                         [bipartite_graph(N, 3, 0.4) for _ in range(n_iter)] + 
                         [bipartite_graph(N, 5, 0.4) for _ in range(n_iter)])
detour_prob, detour_amp = divide([detour_graph(N, 2) for _ in range(n_iter)] +
                                 [detour_graph(N, 4) for _ in range(n_iter)] + 
                                 [detour_graph(N, 6) for _ in range(n_iter)] + 
                                 [detour_graph(N, 8) for _ in range(n_iter)])

# Plot graph
plt.scatter(star_amp, star_prob, marker='^', label='Modified star', color = 'Orange')
plt.scatter(bp_amp, bp_prob, marker='s', label='Bipartite', color = 'Red')
plt.scatter(er_amp, er_prob, marker='P', label='Erdos Renyi', color='Green')
plt.scatter(rg_amp, rg_prob, marker='*', label='Random geometric', color='skyblue')
plt.scatter(detour_amp, detour_prob, marker='o', label='Detour', color='blue')
plt.xlabel('Amplification factor')
plt.ylabel('Fixation probability')
plt.legend()
plt.savefig("figure1.pdf", dpi=300)