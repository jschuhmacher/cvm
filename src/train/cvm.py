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
File: cvm.py

Implementation of the Core Vector Machine.
'''

import common.data
import common.model
import common.profile
import classify.predictor
import numpy as np
import scipy as sp
import scipy.sparse as spsparse
import logging

class CVM( object ):
    ''' Implementation of the Core Vector Machine described in the 2006 paper by 
        Tsang et al. "Core Vector Machines: Fast SVM Training on Very Large Data 
        Sets" published in The Journal of Machine Learning Research.
    '''

    ''' The core set: S
        A list of indices to the dataset
    '''
    coreset = np.array( [], dtype='i' )

    ''' The radius of the ball: R '''
    radius = None

    ''' The model this classifier will learn '''
    profile = None


    ''' Class labels translated to {-1, +1} for the current class '''
    labels = np.array( [], dtype='f' )

    ''' Data distribution for this class '''
    nr_class = 0
    nr_other = 0

    ''' Lagrange multipliers '''
    alphas = np.array( [], dtype='f' )

    ''' Acceptable approximation for the MEB '''
    epsilon = 1.0e-6

    dataset = None

    def __init__( self, model, dataset ):
        ''' '''
        self.model = model
        self.dataset = dataset

        # Count class distribution
        for label in self.dataset.classes:
            if label == model.class_name:
                self.labels.append( 1 )
                self.nr_class += 1
            else:
                self.labels.append( -1 )
                self.nr_other += 1
        logging.debug( "Class distribution for classes <{0!s}>:%{1!s}\
            <other>:{2!s}.".format( model.class_name, self.nr_class, self.nr_other ) )

    def initialise( self, class_profile ):
        ''' Initialise S_0, c_0 and R_0
        
            TODO: Improve initialising the core set by choosing 
                  a point z_0, choosing a second point z_a that is furthest 
                  away from z_0 and a point z_b that is furthest away from z_a.
                  Points z_a and z_b will be the initial set.
        '''
        # Initialise the coreset
        z_a = z_b = random.randint( 1, self.dataset.size() )

        # Make sure the points are from different classes
        while self.dataset.size() > 1 and \
                ( z_b == z_a or self.labels[z_a] == self.labels[z_b] ):
            z_b = random.randint( 1, self.dataset.size() )

        self.coreset.resize( ( 1, self.coreset.size() + 2 ) )
        self.coreset[self.coreset.size() - 2] = z_a
        self.coreset[self.coreset.size() - 1] = z_b

        # initialise the center, this is a bit of a problem as the center is no
        # explicit point, it is a convex combination of \tilde{\Phi}(x)_i's

        # Initialise the Lagrange multipliers to 1 / nr_samples
        self.alphas.resize( ( 1, self.coreset.size() + 2 ) )
        self.alphas[self.coreset.size() - 2] = .5
        self.alphas[self.coreset.size() - 1] = .5

    def hasRemainingPoint ( self, class_profile ):
        ''' Check if there are any remaining points present that fall outside
            of the ( 1 + e ) - approximation B( c_t, ( 1 + e ) R_t )
        '''
        pass

    def train ( self ):
        ''' Learns ( w, b ) from training examples with 2 classes
        '''
        self.initialise()
        iteration = 0
        while( self.hasRemainingPoint() ):
            self.updateCoreSet()
            self.solveQP()
            self.updateBall()
            iteration += 1


    def updateBall( self ):
        ''' Update the MEB '''
        pass

    def computeDistanceCenterToXl( self, x_l ):
        ''' Computes the squared distance from a point x_l to the center of 
            the coreset 
        '''
        distance = 0
        distance2 = 0

        for i in range( 0, self.coreset.size() ):
            for j in range( 0, self.coreset.size() ):
                distance += self.alphas[i] * self.alphas[j] * \
                    self.cvm_kernel( i, j )

            distance2 += self.alphas[i] * self.cvm_kernel( i, x_l ) + \
                self.cvm_kernel( x_l, x_l )

        return distance - distance2

    def computeMaximumDistanceToSampling( self, distance, distance_i ):
        ''' '''
        pass

    def computeRadiusSquared ( self ):
        ''' Compute the square of the radius '''

        pass

    def updateCoreSet( self ):
        ''' Add new points to the coreset '''
        pass

    def cvm_kernel ( self, i, j ):
        ''' The CVM kernel function 
            
            TODO (in case of slowness): Implement a caching scheme
        '''
        return self.labels[i] * self.labels[j] * self.kernel( i, j ) + \
            self.labels[i] * self.labels[j] + ( self.dataset.data[i, j] / C )

    def kernel( self, i, j ):
        ''' The actual kernel function, only dot product for now

            TODO (someday): Implement other kernels as well
            TODO (in case of slowness): Implement a caching scheme
        '''
        return np.linalg.dot( self.dataset.data[i, :],
                              self.dataset.data[j, :] )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
