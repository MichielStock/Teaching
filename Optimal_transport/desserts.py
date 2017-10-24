"""
Created on Sunday 22 October 2017
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Create the dessert distribution example
"""

import pandas as pd
from optimal_transport import compute_optimal_transport, OptimalTransport

# preferences
preferences = pd.DataFrame([
    [2, 2, 1, 0, 0],
    [0, -2, -2, -2, 2],
    [1, 2, 2, 2, -1],
    [2, 1, 0, 1, -1],
    [0.5, 2, 2, 1, 0],
    [0, 1, 1, 1, -1],
    [-2, 2, 2, 1, 1],
    [2, 1, 2, 1, -1]
], index=['Bernard', 'Jan', 'Willem', 'Hilde', 'Steffie', 'Marlies', 'Tim', 'Wouter'])

preferences.columns = ['merveilleux', 'eclair', 'chocolate mousse', 'bavarois', 'carrot cake']

M = - preferences.values

# prortions per person
portions_per_person = pd.DataFrame([[3],
                                    [3],
                                    [3],
                                    [4],
                                    [2],
                                    [2],
                                    [2],
                                    [1]],
                    index=['Bernard', 'Jan', 'Willem', 'Hilde', 'Steffie',
                                            'Marlies', 'Tim', 'Wouter'])
# quantities
quantities_of_dessert = pd.DataFrame([  [4],
                                        [2],
                                        [6],
                                        [4],
                                        [4]],
                                        index=['merveilleux', 'eclair', 'chocolate mousse',
                                                   'bavarois', 'carrot cake'])

if __name__ == '__main__':
    pass                                            
