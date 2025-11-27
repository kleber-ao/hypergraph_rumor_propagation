import numpy as np
from src.model.utils import store_incidence_matrix_into_two_dicts, Random_number_generator, unif_to_exp

def rph_gillespie_simulation(incidence_matrix, lambda_contagion_rate, alpha_annihilation_rate, contagion_threshold, annihilation_threshold, initial_infection_size, seed, rng_batchsize):
    '''
    Runs a single Gillespie simulation of the rumor propagation model on hypergraphs.
    Arguments
        - incidence_matrix: scipy.sparse.csr_matrix with rows for nodes and columns for hyperedges
        - lambda_contagion_rate: float for the rate of contagion
        - alpha_annihilation_rate: float for the rate of annihilation
        - contagion_threshold: float between 0 and 1 for how many nodes are necessary to activate a hyperedge contagion process
        - annihilation_threshold: float between 0 and 1 for how many hyperedges are necessary to activate a node annihilation process
        - initial_infection_size: integer for how many hypergraphs are randomly activated at the beginning of the simulation
        - seed: integer for the random number generator seed
        - rng_batchsize: how many random numbers are generated at each batch. Trade-off between speed and memory
    Outputs
        - event_times: list of floats with all events during which something happened
        - compartment_occupancy: how many nodes are in each comparment for each time in event_times
    '''

    ## Initialization

    # network
    N_network_size = incidence_matrix.shape[0]
    M_no_of_hyperedges = incidence_matrix.shape[1]
    edges_incident_to, nodes_belonging_to = store_incidence_matrix_into_two_dicts(incidence_matrix)

    # thresholding
        # there are two threshold rule options:
        # 1st: if theta is in (0,1), threshold counts excess from theta*hyperedge_size or from theta*degree;
        # 2nd: if theta is a natural number, threshold counts excess from theta regardless of hyperedge size or degree;
    alternate_thresholding_for_contagion = False
    alternate_thresholding_for_annihilation = False
    if np.mod(contagion_threshold,1) == 0: # if any theta is natural, turn the flag for alternate thresholding on
        alternate_thresholding_for_contagion = True
    if np.mod(annihilation_threshold,1) == 0:
        alternate_thresholding_for_annihilation = True

    # output variables
    event_times = []
    X = [] # susceptible
    Y = [] # spreader
    Z = [0] # stifler

    # control variables

    # the states of nodes is controlled through an array of integers where each position can only be:
    #   0: susceptible
    #   1: spreader
    #   2: stifler
    node_states = np.zeros(N_network_size, dtype=np.int32)

    # it is also useful to keep track of whether hyperedges have activated or not
    #   0: did not activate yet
    #   1: has activated at some point
    hyperedge_has_activated = np.zeros(M_no_of_hyperedges, dtype=np.int32)

    contagion_processes = set() # set with hyperedge ids
    annihilation_processes = set() # set with node ids

    rng = Random_number_generator(seed,rng_batchsize)

    # seed infection
    t_current_time = 0
    event_times.append(t_current_time)

    # trigger the initial infection
    for i in range(initial_infection_size):
        infected_hyperedge = np.int32(np.round(rng.get()*(M_no_of_hyperedges-1)))

        if hyperedge_has_activated[infected_hyperedge] == 0:
            infected_from_seed = nodes_belonging_to[infected_hyperedge]
            if not (node_states[infected_from_seed]==1).all():
                node_states[infected_from_seed] = 1
            else:
                while (node_states[infected_from_seed]==1).all():
                    infected_hyperedge = np.int32(np.round(rng.get()*(M_no_of_hyperedges-1)))
                    infected_from_seed = nodes_belonging_to[infected_hyperedge]
                node_states[infected_from_seed] = 1
        
        hyperedge_has_activated[infected_hyperedge] = 1
    
    # find initial contagion processes
    for v in np.where(node_states==1)[0]:
        for e in edges_incident_to[v]:
            number_of_spreaders_in_e = len(np.where(node_states[nodes_belonging_to[e]]==1)[0])
            hyperedge_size = len(nodes_belonging_to[e])
            if hyperedge_has_activated[e] == 0:
                # threshold check
                if not alternate_thresholding_for_contagion:
                    critical_mass = np.int32(np.round(hyperedge_size*contagion_threshold)) # critical mass is always an integer
                    if critical_mass < 1: # it has to be at least a single infected node
                        critical_mass = 1
                else:
                    critical_mass = contagion_threshold
                if number_of_spreaders_in_e >= critical_mass:
                    contagion_processes.add(e)

    # find initial annihilation processes
    for v in np.where(node_states==1)[0]:
        if v not in annihilation_processes:
            activated_hyperedges_of_v = np.sum(hyperedge_has_activated[edges_incident_to[v]])
            degree_of_v = len(edges_incident_to[v])
            # check for annihilation threshold
            
            if not alternate_thresholding_for_annihilation:
                critical_mass = np.int32(np.round(degree_of_v*annihilation_threshold))
                if critical_mass < 1:  # it has to be at least a single infected hyperedge
                    critical_mass = 1
            else:
                critical_mass = annihilation_threshold
            
            if activated_hyperedges_of_v >= critical_mass:
                annihilation_processes.add(v)

    # update output variables with the initial infection    
    Y.append(len(node_states[node_states==1])/N_network_size)
    X.append(len(node_states[node_states==0])/N_network_size)

    # initial rates for the Gillespie simulation
    total_contagion_rate = lambda_contagion_rate*(len(contagion_processes))
    total_annihilation_rate = alpha_annihilation_rate*len(annihilation_processes)
    total_rate = total_contagion_rate + total_annihilation_rate

    ## Event iteration
    while total_rate > 0:
        t_current_time += unif_to_exp(rng.get(),total_rate) # all event time random drawing should have this shape
        event_times.append(t_current_time)
        
        # 1. annihilation event
            # 1.1 remove process from annilation_processes
            # 1.2 change node state from spreader to stifler
            # 1.3 check if contagion processes need to be removed
        if rng.get()*total_rate < total_annihilation_rate:
            random_process_index = np.int32(np.round(rng.get()*(len(annihilation_processes)-1)))
            new_stifler = tuple(annihilation_processes)[random_process_index]
            # 1.1
            annihilation_processes.remove(new_stifler)
            # 1.2
            node_states[new_stifler] = 2
            # 1.3
            for e in edges_incident_to[new_stifler]:
                if e in contagion_processes:
                    number_of_spreaders_in_e = len(np.where(node_states[nodes_belonging_to[e]]==1)[0])
                    hyperedge_size = len(nodes_belonging_to[e])
                    # check for contagion threshold
                    if not alternate_thresholding_for_contagion:
                        critical_mass = np.int32(np.round(hyperedge_size*contagion_threshold))
                        if critical_mass < 1:
                            critical_mass = 1
                    else:
                        critical_mass = contagion_threshold

                    if number_of_spreaders_in_e < critical_mass:
                        contagion_processes.remove(e)

        # 2. contagion event
            # 2.1 remove process from contagion_processes
            # 2.2 change hyperedge tracking to store it has been activated
            # 2.3 change node states from ignorants to spreaders
            # 2.4 check if there are new contagion processes
            # 2.5 check if there are new annihilation processes
        else:
            random_process_index = np.int32(np.round(rng.get()*(len(contagion_processes)-1)))
            newly_activated_hyperedge = tuple(contagion_processes)[random_process_index]
            # 2.1
            contagion_processes.remove(newly_activated_hyperedge)
            # 2.2
            hyperedge_has_activated[newly_activated_hyperedge] = 1
            # other items
            for v in nodes_belonging_to[newly_activated_hyperedge]:
                if node_states[v] == 0:
                    # 2.3
                    node_states[v] = 1
                    # 2.4
                    for e in edges_incident_to[v]:
                        if e not in contagion_processes and hyperedge_has_activated[e] == 0:
                            number_of_spreaders_in_e = len(np.where(node_states[nodes_belonging_to[e]]==1)[0])
                            hyperedge_size = len(nodes_belonging_to[e])
                            # check for contagion threshold
                            if not alternate_thresholding_for_contagion:
                                critical_mass = np.int32(np.round(hyperedge_size*contagion_threshold))
                                if critical_mass < 1:
                                    critical_mass = 1
                            else:
                                critical_mass = contagion_threshold

                            if number_of_spreaders_in_e >= critical_mass:
                                contagion_processes.add(e)
                # 2.5
                if v not in annihilation_processes:
                    activated_hyperedges_of_v = np.sum(hyperedge_has_activated[edges_incident_to[v]])
                    degree_of_v = len(edges_incident_to[v])
                    # check for annihilation threshold
                    
                    if not alternate_thresholding_for_annihilation:
                        critical_mass = np.int32(np.round(degree_of_v*annihilation_threshold))
                        if critical_mass < 1:
                            critical_mass = 1
                    else:
                        critical_mass = annihilation_threshold
                    
                    if activated_hyperedges_of_v >= critical_mass:
                        annihilation_processes.add(v)

        # update output variables within iteration
        X.append(len(node_states[node_states==0])/N_network_size)
        Y.append(len(node_states[node_states==1])/N_network_size)
        Z.append(len(node_states[node_states==2])/N_network_size)

        # update rates
        total_contagion_rate = lambda_contagion_rate*len(contagion_processes)
        total_annihilation_rate = alpha_annihilation_rate*len(annihilation_processes)
        total_rate = total_contagion_rate + total_annihilation_rate

    compartment_occupancy = (X,Y,Z)
    return event_times, compartment_occupancy