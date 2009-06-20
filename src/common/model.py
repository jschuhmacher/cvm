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

''' 
File: model.py

moduledocs
'''

import math
import numpy as np
from scipy import sparse

# 3rd-party
import psyco
psyco.full()

class Model( object ):
    ''' classdocs
    '''
    class_name = None
    support_vectors = None
    support_vectors_cache = None
    support_vector_labels = None
    nr_svs = 0
    alphas = None
    b = None

    def __init__( self, alphas, b, class_name, nr_svs, support_vector_labels,
                  support_vectors ):
        self.class_name = class_name
        self.support_vectors = support_vectors

        self.support_vectors_cache = support_vectors.copy()
        for i in range( self.nr_svs ):
            self.support_vectors_cache[i] = self.support_vectors[i] * \
                                            self.support_vectors[i].T

        self.support_vector_labels = support_vector_labels
        self.nr_svs = nr_svs
        self.alphas = alphas
        self.b = b

    def predict( self, x ):
        ''' Predict class for point x. '''
        score = 0.0

        for i in range( self.nr_svs ):
            score += ( self.alphas[i, 0] * \
                       self.support_vector_labels[i, 0] * \
                       self.kernel( i, x )[0, 0] )
        score += self.b

        return score

    def kernel( self, i, x ):
        ''' 
        The actual kernel function, only dot product for now
        '''
        #k_ij = ( self.support_vectors[i] * x.T ).todense()
        #k_ii = ( self.support_vectors[i] * self.support_vectors[i].T ).todense()
        #k_jj = ( x * x.T ).todense()
        #return ( k_ij / ( np.math.sqrt( k_ii ) * np.math.sqrt( k_jj ) ) )

        return ( ( self.support_vectors[i] * x.T ).todense() / ( 
                    np.math.sqrt( ( self.support_vectors[i] * self.support_vectors[i].T ).todense() ) *
                    np.math.sqrt( ( x * x.T ).todense() ) ) )
