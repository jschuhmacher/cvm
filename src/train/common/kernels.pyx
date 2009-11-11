#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sw=4 expandtab:

################################################################################
# Copyright (c) 2009, Jelle Schühmacher <j.schuhmacher@student.science.ru.nl>
#
# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################################
cimport numpy as np
cimport cython

import logging
import numpy as np
import scipy as sp
import scipy.sparse as sparse

@cython.boundscheck( False )
def norm_dot_kernel ( x_i, x_j ):
    assert( sparse.isspmatrix_csr( x_i ) and sparse.isspmatrix_csr( x_j ) )
    cdef double k_ij = ( x_i * x_j.T )[0, 0]
    cdef double k_ii = ( x_i * x_i.T )[0, 0]
    cdef double k_jj = ( x_j * x_j.T )[0, 0]

    cdef double res = ( k_ij / ( np.math.sqrt( k_ii ) * np.math.sqrt( k_jj ) ) )
    return res

@cython.boundscheck( False )
def dot_kernel ( x_i, x_j ):
    assert( sparse.isspmatrix_csr( x_i ) and sparse.isspmatrix_csr( x_j ) )
    cdef double res = ( x_i * x_j.T )[0, 0]
    return res

@cython.boundscheck( False )
def rbf_kernel ( x_i, x_j, double GAMMA ):
    assert( sparse.isspmatrix_csr( x_i ) and sparse.isspmatrix_csr( x_j ) )
    cdef double res = np.exp( -GAMMA * ( ( x_i * x_i.T )[0,0] - 2.0 * ( x_i * x_j.T )[0,0] + ( x_j * x_j.T )[0,0] ) )
    return res

@cython.boundscheck( False )
def cvm_kernel ( double label_i, double label_j, x_i, x_j, unsigned int i,
                 unsigned int j, kernel, GAMMA, C ):
    ''' The CVM kernel function
    '''
    assert( sparse.isspmatrix_csr( x_i ) and sparse.isspmatrix_csr( x_j ) )
    cdef double d_ij = 0.0
    if i == j:
        d_ij = 1.0

    cdef double t_ij = label_i * label_j
    cdef double res = t_ij * kernel( x_i, x_j, GAMMA ) + t_ij + ( d_ij / C )
    return  res
