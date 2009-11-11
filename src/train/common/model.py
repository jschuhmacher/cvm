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
import common.kernels as kernels

class Model( object ):

    def __init__( self, alphas, b, class_name, nr_svs, support_vector_labels,
                  support_vectors, kernel_func, gamma, ro, C ):
        self.__class_name = class_name
        self.__support_vectors = support_vectors.copy()
        self.__support_vector_labels = support_vector_labels.copy()
        self.__nr_svs = nr_svs
        self.__alphas = alphas.copy()
        self.__b = b
        self.__kernel_func = kernel_func
        self.__gamma = gamma
        self.__ro = ro
        self.__C = C

    def predict( self, x ):
        ''' Predict class for point x. '''
        score = 0.0
        for i in range( self.__nr_svs ):
            score += ( self.__alphas[i, 0] * \
                       self.__support_vector_labels[i, 0] * \
                       self.__kernel( i, x ) )

        score += self.__b
        return score

    def __kernel( self, i, x ):
        ''' 
        The actual kernel function, only dot product for now
        '''
        return self.__kernel_func( x, self.__support_vectors[i], self.__gamma )
