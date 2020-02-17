# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:03:38 2018

@author: Knowhow
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:16:49 2018

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

@author: Maximilian David Flohr
"""
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)

def calc_density(
                np.ndarray[np.float64_t, ndim=1] position,
                unsigned int ncells,
                double L):
    """ Calculate charge density given particle positions

    Input
      position  - Array of positions, one for each particle
                  assumed to be between 0 and L
      ncells    - Number of cells
      L         - Length of the domain

    Output
      density   - contains 1 if evenly distributed
    """
    # This is a crude method and could be made more efficient
    cdef np.ndarray[np.float64_t, ndim=1] density = np.zeros(ncells,
                                                       dtype=np.float64)
    cdef unsigned int nparticles = position.shape[0]
    cdef unsigned int pLower
    cdef unsigned int i
    cdef double offset
    cdef double dx = L / ncells
    cdef double p
    cdef int i_max= position.shape[0]
    for i in range(0,i_max):
        p = position[i]/dx
        pLower = int(p)
        offset = p - pLower
        density[pLower] += 1. - offset
        density[(pLower + 1) % ncells] += offset
    for i in range(0,ncells):
        density[i] *= <double>ncells / <double>nparticles
    return density



