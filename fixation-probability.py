import numpy as np
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

# Effect function b
def b(c, s, h, Sp, alpha):
    if s==0: # Linear rewards function just returns the concentration
        return c/(Sp/alpha)
    v_c = 1/(1 + np.exp(-s*(c - h))) # v at given concentration
    vmin = 1/(1 + np.exp(-s*(0 - h))) # v at min concentration
    vmax = 1/(1 + np.exp(-s*(Sp/alpha - h))) # v at max concentration
    b_c = (v_c - vmin)/(vmax - vmin)
    return b_c

# Calculate the fitness of the current state
def network_fit(pop_state, K_inv, Sp, alpha, s, h, k, omega):
    f = np.zeros(len(pop_state)) # Initialize fitness vector
    flipped_state = np.array([1 - state for state in pop_state])
    
    # Define production vector for pop_state
    S_state = Sp * flipped_state
    # Solve for steady-state concentrations
    C = np.matmul(K_inv, S_state)
    
    # Calculate payoff vector
    payoff = b(C, s, h, Sp, alpha) - k * flipped_state
    # Calculate fitness-payoff mapping
    f = 1 + omega * payoff
    
    return f

# Calculate the fixation probability when starting from a single mutant for a graph
def pfix_diffusion(G, D, Sp, alpha, s, h, k, omega):
    N = len(list(G)) # Number of nodes in graph
    # Calculating K_inv
    L = nx.laplacian_matrix(G).toarray() # Calculate Laplacian
    K = np.add(D * L, np.diag([alpha]*N))
    K_inv = np.linalg.inv(K)
    
    # Enumerate and store all 2^N possible states
    states = []
    n_mut = []
    for i in range(N+1):
        sub_states = [set(j) for j in combinations(range(N), i)] # All possible graphs with i mutants
        states += sub_states
        n_mut += [i] * len(sub_states)
        
    vs = len(states) # The number of possible graphs
    inv_deg = 1/ np.array(list(dict(nx.degree(G)).values())) # Stores the inverse of the degree for each node (used in replacement probabilities
    
    # Iterate over all possible state transitions and population the transition matrix
    M = np.zeros([vs, vs])
    for i in range(vs):
        for j in range(i+1, vs):
            d = states[j].difference(states[i]) # Added mutant node indices
            d_len = len(states[j]) - len(states[i]) # Number of added mutants
            if len(d) == 1 and d_len == 1: # Transition only possible when they differ by 1
                nn = set(G.neighbors(list(d)[0]))
                nn_int = nn.intersection(states[i]) # Mutant nodes neighboring d in state i
                nn_diff = nn.difference(states[i]) # Wild type nodes neighboring d in state i
                
                # Getting state vectors
                pop_state_i = np.array([1 if k in states[i] else 0 for k in range(N)])
                pop_state_j = np.array([1 if k in states[j] else 0 for k in range(N)])
                # Getting fitness vector for each state
                fitness_i = network_fit(pop_state_i, K_inv, Sp, alpha, s, h, k, omega)
                total_i = sum(fitness_i)
                fitness_j = network_fit(pop_state_j, K_inv, Sp, alpha, s, h, k, omega)
                total_j = sum(fitness_j)     
            
                # Probability of reproducing and replace wildtypeand mutant for reproduction
                wt_prob = np.sum(fitness_j[list(nn_diff)]/total_j * inv_deg[list(nn_diff)])
                mut_prob = np.sum(fitness_i[list(nn_int)]/total_i * inv_deg[list(nn_int)])
                
                # Transition probabilities are reproduction of mutant or wild-type times the replacement probability in each neighboring node
                if len(nn_int) == 0:
                    # Only j -> i is possible if all neighbors are wild-type
                    M[j, i] = wt_prob
                    
                elif len(nn_int) == len(nn):
                    # Only i -> j is possible if all neighbors are mutant
                    M[i, j] = mut_prob
                else:
                    # Both are possible
                    M[i, j] = mut_prob
                    M[j, i] = wt_prob
        
    # Ensure that all rows sum to 1 by controlling diagonal element
    z = np.sum(M, 1)
    for i in range(vs):
        M[i,i] = 1 - z[i]
    
    # Get submatrices in canonical form
    Q = M[1:-1, 1:-1] # Transition probabilities between transient states
    R = np.vstack([M[1:-1, 0], M[1:-1, -1]]).T # Transition probabilties from transient to absorbing states
    N_mat = np.linalg.inv((np.eye(vs - 2) - Q)) # Fundamental matrix
    B = N_mat @ R # Absorption probability matrix
    pfix_mat = np.mean(B[:N, 1]) # Absorption probability from states with 1 mutant    
    
    # return B[:N, 1] # Used if we are concerned with the fixation probability from certain nodes (i.e Figure 2b)
    return pfix_mat
