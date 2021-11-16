import scipy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import itertools


def PreferentialAttachment(k, n_nodes):
    #starting with a complete graph
    G = nx.complete_graph(k+1)
    
    for t in range(k+1, n_nodes):
        if k % 2 != 0: #odd
            #our measure for selecting the degree
            c = int(np.ceil(k/2)) if t % 2 == 0 else int(np.floor(k/2))
        else: #even
            c = int(k/2)
        degrees = [d for n, d in G.degree()] 
        degrees /= np.sum(degrees)
        neighbors = np.random.choice(list(G.nodes), p=degrees, size=c, replace=False)
        #for every time t  we select the node with stochatisc rule to connect
        G.add_node(t)
        for n in neighbors:
            G.add_edge(t, n)

    return G


# S=0 I=1 R=2
def Infection(node, beta, neighbors, state):
    #here since we already have the probability function we dont need to create lambda
    m = list(state[neighbors] == 1).count(True)
    p_infection  = 1 - (1 - beta)**m
    return np.random.choice([0, 1], p = [1-p_infection, p_infection])


def InfectionV(node, beta, neighbors, state, vaccinated):
    #here since we already have the probability function we dont need to create lambda
    m = 0
    neighbors = neighbors[vaccinated[neighbors] == 0]
    m = list(state[neighbors] == 1).count(True)
    p_infection  = 1 - (1 - beta)**m
    return np.random.choice([0, 1], p = [1-p_infection, p_infection])

def Recovery(node, ro):
    return np.random.choice([1, 2], p = [1-ro, ro])

def Vaccinated(W, vaccinated, amount):
    not_vaccinated = np.argwhere(vaccinated == 0).flatten()
    size = int(len(vaccinated)*amount/100)
    vaccinated_nodes = np.random.choice(not_vaccinated, size = size, replace = False)
    return vaccinated_nodes


def Epidemicw(W, n = 500, beta = 0.3, ro = 0.7, init_inf_nodes = 10, num_simul = 100, steps = 15):
    stats_t = []
    inf_stats_t = []
    
    for sim in tqdm(range(num_simul)):

        #creation of intial config
        state = np.zeros(n, dtype = np.int8)
        #insertion of infected nodes
        initial_infected_nodes = np.random.choice(range(n), size = init_inf_nodes, replace = False)
        state[initial_infected_nodes] = 1
        states = []
        #insertion of others nodes
        states.append([n - init_inf_nodes, init_inf_nodes, 0])
        
        #generation of new states
        new_state = np.copy(state)
        new_inf_nodes = []
        new_inf_nodes.append(0)
        
        #creation of time frames and 
        for t in range(steps):
            infected_nodes = 0
            for node in range(n):
                if state[node] == 0: # S to I
                    new_state[node] = Infection(node, beta, W[node].indices, state)
                    if new_state[node] == 1:
                        infected_nodes += 1

                if state[node] == 1: # I to R
                    new_state[node] = Recovery(node, ro)

            #record each stats in time frame
            new_inf_nodes.append(infected_nodes)
            n_S = list(new_state == 0).count(True)
            n_I = list(new_state == 1).count(True)
            n_R = list(new_state == 2).count(True)
            states.append([n_S, n_I, n_R])
            state = np.copy(new_state)
        stats_t.append(states.copy())
        inf_stats_t.append(new_inf_nodes.copy())
        
        
    return stats_t, inf_stats_t

def Epidemic(k=6 , n = 500, beta = 0.3, ro = 0.7, init_inf_nodes = 10, num_simul = 100, steps = 16, vaccine = 0, referencev=0):
    stats_t = []
    inf_stats_t = []
    vaccine_schedule = [5, 9, 16, 24, 32, 40, 47, 54, 59, 60, 60, 60, 60, 60, 60, 60]
    reference_vac = [1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0]
    vaccine = vaccine_schedule
    referencev=reference_vac
    #graph generator for omptimization case
    G = PreferentialAttachment(k = k, n_nodes = n)
    W = nx.adjacency_matrix(G)
    
    if vaccine == 0:
        for sim in tqdm(range(num_simul)):

            #creation of intial config
            state = np.zeros(n, dtype = np.int8)
            #insertion of infected nodes
            initial_infected_nodes = np.random.choice(range(n), size = init_inf_nodes, replace = False)
            state[initial_infected_nodes] = 1
            states = []
            #insertion of others nodes
            states.append([n - init_inf_nodes, init_inf_nodes, 0])

            #generation of new states
            new_state = np.copy(state)
            new_inf_nodes = []
            new_inf_nodes.append(0)

            #creation of time frames and 
            for t in range(steps):
                infected_nodes = 0
                for node in range(n):
                    if state[node] == 0: # S to I
                        new_state[node] = Infection(node, beta, W[node].indices, state)
                        if new_state[node] == 1:
                            infected_nodes += 1

                    if state[node] == 1: # I to R
                        new_state[node] = Recovery(node, ro)

                #record each stats in time frame
                new_inf_nodes.append(infected_nodes)
                n_S = list(new_state == 0).count(True)
                n_I = list(new_state == 1).count(True)
                n_R = list(new_state == 2).count(True)
                states.append([n_S, n_I, n_R])
                state = np.copy(new_state)
            stats_t.append(states.copy())
            inf_stats_t.append(new_inf_nodes.copy())
            
    else:
        for sim in tqdm(range(num_simul)):
    
            state = np.zeros(n, dtype = np.int8)
            initial_infected_nodes = np.random.choice(range(n), size = init_inf_nodes, replace = False)
            state[initial_infected_nodes] = 1
            states = []
            states.append([n - init_inf_nodes, n - init_inf_nodes, init_inf_nodes, 0, 0, 0])
            vaccinated = np.zeros(n, dtype = np.int8)
            new_state = np.copy(state)
            new_inf_nodes = []
            for t in range(steps):
                # Vaccinations
                amount = max(0, vaccine[t]-vaccine[t-1])
                if amount > 0:
                    vaccinated_nodes = Vaccinated(W, vaccinated, amount)
                    vaccinated[vaccinated_nodes] = 1
                infected_nodes = 0

                for node in range(n):
                    if state[node] == 0 and vaccinated[node] != 1: # S
                        new_state[node] = InfectionV(node, beta, W[node].indices, state, vaccinated)
                        if new_state[node] == 1:
                            infected_nodes += 1

                    if state[node] == 1: # I
                        new_state[node] = Recovery(node, ro)

                new_inf_nodes.append(infected_nodes)
                n_SV = list(vaccinated[new_state == 0] == 0).count(True) #S after I
                n_S = list(new_state == 0).count(True) # S by vaccine
                n_I = list(new_state == 1).count(True)
                n_R = list(new_state == 2).count(True)
                n_V = list(vaccinated == 1).count(True)
                
                states.append([n_SV, n_S, n_I, n_R, n_V, amount])
                state = np.copy(new_state)
            stats_t.append(states.copy())
            inf_stats_t.append(new_inf_nodes.copy())
    mean_inf_stats_t = np.mean(inf_stats_t, axis = 0)
    rmse = mean_squared_error(mean_inf_stats_t, referencev, squared=False)  
    return stats_t, inf_stats_t, mean_inf_stats_t, rmse


def GradientOptimizer(k=10, beta=0.3, ro=0.6, step_k=2, step_beta=0.1, step_ro=0.1):
    
    best_rmse = -1
    test = 0
    mult = 0.5
    threshold = 1e-5
    doupdate = 0
    ref_vac = [1, 1, 3, 5, 9, 17, 32, 32, 17, 5, 2, 1, 0, 0, 0, 0]

    while True:
        local_best = -1

        changed = False
        print(f"Starting Parameters with k={k:.3g}, beta={beta:.3g}, ro={ro:.3g}")

        choices = [set([max(0,beta-(mult**test)*step_beta), beta, min(1,beta+(mult**test)*step_beta)]), 
                   set([max(0,int(k-(mult**test)*step_k)), k, int(k+(mult**test)*step_k)]), 
                   set([max(0,ro-(mult**test)*step_ro), ro, min(1,ro+(mult**test)*step_ro)]) ]
        params = list(itertools.product(*choices))
        for i, param in enumerate(params):
            beta_sim, k_sim, ro_sim = param

            simulations_states, inf_stats_t, mean_inf, rmse = Epidemic(k = k_sim, beta = beta_sim, ro = ro_sim)

            if local_best == -1 or rmse < local_best:
                    best_local_mean = mean_inf
                    beta_local_best, k_local_best, ro_local_best = beta_sim, k_sim, ro_sim
                    local_best = rmse
                    local_best_states = simulations_states

        # Update global best
        if local_best < best_rmse or best_rmse == -1:
            best_states = local_best_states
            best_rmse = local_best
            best_mean = best_local_mean
            beta_best, k_best, ro_best = beta_local_best, k_local_best, ro_local_best
            changed = True

        if changed:
            print(f"Best RMSE located : {best_rmse:.3g} at: beta={beta_best:.3g} k={k_best:.3g} ro={ro_best:.3g}")  
            plt.figure(figsize=(10,10))
            plt.grid()
            plt.plot(range(16), best_mean, label='Model Approximation')
            plt.plot(range(16), ref_vac, label='Reference')
            plt.legend(title='Nodes', loc='upper left')
            plt.show()
            doupdate = 0
        else:
            doupdate += 1

        # Check if min hasn't moved
        if abs(beta_local_best - beta) < threshold and abs(k_local_best - k) < threshold and abs(ro_local_best - ro) < threshold and test <= 1:
            print("updating precision")
            test += 1
        # Move to new min
        elif abs(beta_local_best - beta) > threshold or abs(k_local_best - k) > threshold or abs(ro_local_best - ro) > threshold:
            print(f"Moving Local Optima")
            beta, k, ro = beta_local_best, k_local_best, ro_local_best
            test = 0
        # Termination conditions
        if test > 1 or doupdate >= 10:
            print("EXIT")
            break

    return beta_best, k_best, ro_best, best_states

