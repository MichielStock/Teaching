"""
Created on Wed Apr 15 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of a HMM with Viterbi tracking for didactic purposes
"""

import numpy as np

def viterbi(seq_int, emission_probs, transition_probs,
        starting_probs):
    '''
    Generates the dynamic programming matrix for a sequence of symbols
    INPUTS:
        - seq_int: list of symbol indices
        - emission_probs: emission_probabilities
        - transition_probs: transition probabilities
        - starting_probs: starting probabilities
    '''
    n_states = transition_probs.shape[0]
    length_seq = len(seq_int)
    dyn_prog_matrix = np.zeros((n_states, length_seq))
    pointers = np.zeros((n_states, length_seq), dtype=int)
    # initialise the matrix
    for i in range(n_states):
        # log of prob starting at this position given first symbol
        dyn_prog_matrix[i, 0] = np.log(starting_probs[i]) +\
                np.log(emission_probs[seq_int[0], i])
    # now fill the rest
    for i, symb in enumerate(seq_int[1:]):
        for state in range(n_states):
            log_probs = []
            for prev_state in range(n_states):
                log_probs.append( dyn_prog_matrix[prev_state, i] +\
                    np.log(transition_probs[prev_state, state]) +\
                    np.log(emission_probs[symb, state]))
            dyn_prog_matrix[state, i+1] = np.max(log_probs)
            pointers[state, i+1] = np.argmax(log_probs)
    return backtracking(dyn_prog_matrix, pointers)

def backtracking(dyn_prog_matrix, pointers):
    '''
    Uses backtracking to find the best sequence
    '''
    pos = dyn_prog_matrix.shape[1] - 1
    pointy_finger = np.argmax(dyn_prog_matrix[:, -1])
    seq = [pointy_finger]
    while pos >= 1:
        pos -= 1
        pointy_finger = pointers[pointy_finger, pos]
        seq = [pointy_finger] + seq
    return seq


if __name__=='__main__':

    symbols = ['H', 'T']
    states = ['F', 'B']

    emis_probs = np.array([[0.5, 0.1], [0.5, 0.9]])
    trans_probs = np.array([[0.8, 0.2], [0.2, 0.8]])
    starting_probs = np.array([0.99, 0.01])

    symbols_to_int = lambda seq : np.array(map(lambda s : symbols.index(s), seq), dtype=int)
    states_to_int = lambda seq : np.array(map(lambda s : states.index(s), seq), dtype=int)

    int_to_symbols = lambda ints : ''.join(map(lambda i: symbols[i], ints))
    int_to_states = lambda ints : ''.join(map(lambda i: states[i], ints))

    sequence = 'HTHHTHHTTTHTTTTTTTTTHTTTHTTHTTHTHHHHTTTTTHHHTHHH'*5

    print sequence
    print int_to_states(viterbi(symbols_to_int(sequence), emis_probs,\
            trans_probs, starting_probs))
