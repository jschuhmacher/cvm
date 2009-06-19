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

# System
import logging
import random
import pprint
import math

# 3rd-party
import numpy as np
import scipy as sp
from scipy import sparse
import cvxopt
import cvxopt.solvers
import psyco

# Local
import common.data
import common.model
import classify.predictor

random.seed( 200 )
psyco.full()

class CVM( object ):
    ''' 
    Implementation of the Core Vector Machine described in the 2006 paper 
    by Tsang et al. "Core Vector Machines: Fast SVM Training on Very Large 
    Data Sets" published in The Journal of Machine Learning Research.
    '''

    # Frustrating fact: Python class attributes are global to all
    # instances of that class

    def __init__( self, class_name, dataset, C=1e6, EPSILON=1.0e-6, NR_SAMPLES=59, MAX_ITERS=30 ):
        ''' Size of the set of samples taken at each iteration '''
        self.NR_SAMPLES = NR_SAMPLES

        ''' Acceptable approximation for the MEB '''
        self.EPSILON = EPSILON

        ''' Regularisation constant '''
        self.C = C

        self.MAX_ITERS = MAX_ITERS

        ''' The dataset used for training. '''
        self.dataset = dataset

        ''' The class this classifier will learn. '''
        self.class_name = class_name

        ''' 
        The core set: S
        A list of pointers into the dataset
        '''
        self.coreset = []

        ''' The squared radius of the ball: R^2 '''
        self.radius2 = 0.0

        ''' Class labels, translated to {-1, +1}, for the current class '''
        self.labels = []

        ''' List of pointers to all positive records. '''
        self.pos_ids = []

        ''' List of pointers to all negative records. '''
        self.neg_ids = []

        ''' Data distribution for this class '''
        self.nr_pos = 0
        self.nr_neg = 0

        ''' Should next sample be positive or negative? '''
        self.pos_turn = True

        ''' Lagrange multipliers '''
        self.alphas = np.zeros( ( 2, 1 ), np.float )

        ''' Small cache for the core vector kernel evaluations '''
        self.k_cache = np.zeros( ( 2, 2 ), np.float )

        self.iteration = 0

        # Solver options
        cvxopt.solvers.options['show_progress'] = True

        # Maximum number of iterations 
        cvxopt.solvers.options['maxiters'] = 200

        # Absolute accuracy
        #cvxopt.solvers.options['abstol'] = 1e-20

        # Relative accuracy
        #cvxopt.solvers.options['reltol'] = 0

        # Tolerance for feasibility condition
        #cvxopt.solvers.options['feastol'] = 1e-20
        #cvxopt.solvers.options['refinement'] = 0

        logging.debug( "    Classifier for <{0}> starts with EPSILON: {1}, C: {2}, NR_SAMPLES: {3}".
                       format( self.class_name,
                               self.EPSILON,
                               self.C,
                               self.NR_SAMPLES ) )

        # Count class distribution
        for i in self.dataset.get_indices():
            label = self.dataset.get_label( i )
            if label == self.class_name:
                self.labels.append( 1 )
                self.pos_ids.append( i )
                self.nr_pos += 1
            else:
                self.labels.append( -1 )
                self.neg_ids.append( i )
                self.nr_neg += 1

        logging.debug( "    Class distribution for class <{0}>:{1}, <other>:{2}.".
                       format( self.class_name, self.nr_pos, self.nr_neg ) )

    def train ( self ):
        ''' 
        Learns a linear model from training examples with 2 classes.
        '''
        self.__initialise()
        self.iteration = 0

        while( self.iteration < self.MAX_ITERS ):
            # Find a point that is further away than the radius + epsilon^2
            new_cs = self.__furthest_point( self.radius2 + ( self.EPSILON / 2.0 ) )

            # No more points remain
            if new_cs == None:
                break
            else:
                self.__update_coreset( new_cs )
                self.__solve_qp()
                self.__update_radius2()
                self.iteration += 1

        logging.info( "  Completed training for class <{0}> in {1} iterations".
                      format( self.class_name, self.iteration ) )

        return self.__make_model()

    def __initialise( self ):
        ''' 
        Initialise S_0, c_0 and R_0
        '''
        # Initialise the coreset with a balanced sample
        z_a = self.__next_sample()
        self.__update_coreset( z_a )
        self.alphas[0, 0] = 0.5

        z_b = self.__next_sample()
        self.__update_coreset( z_b )
        self.alphas[1, 0] = 0.5

        #self.__update_alphas()

        # Construct a cache of kernel evaluations of the core vectors
        n = len( self.coreset )
        for i in range( n ):
            for j in range( n ):
                self.k_cache[i, j] = self.__cvm_kernel( i, j )

        # Compute a new radius
        self.__update_radius2()

    def __next_sample( self ):
        ''' 
        Returns an id in the list of datapoints. Tries to return balanced
        random samples when enough data is present.
        '''
        # No points left
        if self.nr_pos <= 0 and self.nr_neg <= 0:
            return None

        if self.nr_pos <= 0:
            self.pos_turn = False
        elif self.nr_neg <= 0:
            self.pos_turn = True

        # Select a random point that has not yet been used
        if self.pos_turn:
            id = self.pos_ids[random.randint( 0, self.nr_pos - 1 )]
        else:
            id = self.neg_ids[random.randint( 0, self.nr_neg - 1 )]

        return id

    def __solve_qp( self ):
        ''' 
        We need to solve the quadratic programme:        

            maximise: - \alpha^T __cvm_kernel \alpha
             \alpha
            
            s.t.: \alpha       >= 0
                  \alpha^T 1    = 1

        using cvxopt. cvxopt solves:
        
            minimise: 0.5 x^T P x + q^T x 
             x
            
            s.t.: Gx    <= h
                  Ax     = b
        
        We transform the qp to:
        
            minimise: 0.5 \alpha^T (2.0 __cvm_kernel) \alpha + 0^T \alpha
             \alpha
            
            s.t.: -1^T \alpha  <= 0
                   1^T \alpha   = 1
        
        and it will hopefully work out.
        '''
        n = len( self.coreset )

        # Fill P[i, j] with kernel evaluations for i and j in the core set
        # cvxopt won't take the numpy.ndarray
        P = cvxopt.matrix( 0.0, ( n, n ) )
        for i in range( n ):
            for j in range( n ):
                P[i, j] = float( self.k_cache[i, j] )

        q = cvxopt.matrix( 0.0, ( n, 1 ) )
        G = cvxopt.matrix( 0.0, ( n, n ) )
        G[::n + 1] = -1.0
        h = cvxopt.matrix( 0.0, ( n, 1 ) )
        A = cvxopt.matrix( 1.0, ( 1, n ) )
        b = cvxopt.matrix( 1.0 )
        x = cvxopt.matrix( 0.0, ( n, 1 ) )

        for i in range( n ):
            x[i, 0] = self.alphas[i, 0]

        logging.info( "  Solving QP for iteration {0}".format( self.iteration ) )

        solution = cvxopt.solvers.qp( P, q, G, h, A, b, None, {'x':x} )

        if solution['status'] != 'optimal':
            logging.debug( '    Could not solve QP problem, results: {0}\n'.
                           format( pprint.pformat( solution ) ) )

        # Update multipliers
        for i in range( n ):
            self.alphas[i, 0] = solution['x'][i, 0]
#
        # logging.debug( '  New Lagrange multipliers:\n' + pprint.pformat( self.alphas ) )

    def __update_radius2( self ):
        ''' 
        Update the MEB.
            
        Calculate the new radius:
            R^2 = \alpha^T diag(K) - \alpha^T K \alpha
        
        as \alpha_i is 0 for non CVs, only entries corresponding to CVs need to 
        be evaluated.
        '''
        self.radius2 = np.dot( self.alphas.T, self.k_cache.diagonal() )[0] - \
                       self.__get_ro()

        logging.info( "  New squared radius: {0:<10.11}".format( self.radius2 ) )

        # The radius is squared, it should be positive
        assert( self.radius2 >= 0.0 )

    def __distance_to_centroid( self, z_l ):
        ''' 
        Computes the squared distance from a point to the center of
        the coreset: 

            ||c_t - \phi(point)||^2 
            = \sum_{i,j} \alpha_i \alpha_j __cvm_kernel(z_i, z_j) 
            - 2 \sum_i \alpha_i __cvm_kernel(z_i, z_l) __cvm_kernel(z_l, z_l)
        
        see Eq. 18 in the paper.
        '''
        cs_size = len( self.coreset )

        distance2 = 0.0
        for i in range( cs_size ):
            distance2 += self.alphas[i, 0] * self.__cvm_kernel( self.coreset[i],
                                                                z_l )

        return ( self.__get_ro() - 2.0 * distance2 + self.__cvm_kernel( z_l,
                                                                        z_l ) )

    def __furthest_point( self, prev_max_distance ):
        ''' 
        Take NR_SAMPLES samples and select the one point that is furthest away 
        from the current center c_t.
        '''
        max_distance_id = None
        max_distance = prev_max_distance

        for i in range( self.NR_SAMPLES ):
            new_id = self.__next_sample()

            # No more points 
            if new_id == None:
                break

            new_distance = self.__distance_to_centroid( new_id )
            if new_distance > max_distance:
                max_distance = new_distance
                max_distance_id = new_id

        return max_distance_id

    def __update_coreset( self, max_dist_id ):
        ''' Add new points to the coreset '''
        assert( max_dist_id != None )
        assert( ( len( self.pos_ids ) + len( self.neg_ids ) ) <= len( self.labels ) )

        if self.labels[max_dist_id] == 1:
            self.nr_pos -= 1
            self.pos_ids.remove( max_dist_id )
        elif self.labels[max_dist_id] == -1:
            self.nr_neg -= 1
            self.neg_ids.remove( max_dist_id )

        self.coreset.append( max_dist_id )
        self.alphas.resize( ( len( self.coreset ), 1 ) )

        # Only do this for the newly added CV, the rest of the matrix can be 
        # copied from the previous iteration.
        n = len( self.coreset )
        if n > 2:
            new_row = np.zeros( ( 1, n ), np.float )
            new_column = np.zeros( ( n - 1, 1 ), np.float )

            for i in range( n ):
                if i < ( n - 1 ):
                    new_column[i, 0] = self.__cvm_kernel( i, n - 1 )
                new_row[0, i] = self.__cvm_kernel( n - 1, i )

            self.k_cache = np.hstack( [self.k_cache, new_column] )
            self.k_cache = np.vstack( [self.k_cache, new_row] )
        else:
            for i in range( n ):
                for j in range( n ):
                    self.k_cache[i, j] = self.__cvm_kernel( i, j )

        self.pos_turn = not self.pos_turn

    def __make_model( self ):
        ''' Store results in the model '''

        logging.info( '  Creating model' )
        nr_cvs = len( self.coreset )
        alphas = sparse.lil_matrix( ( nr_cvs, 1 ) )
        support_vectors = sparse.lil_matrix( ( nr_cvs, self.dataset.get_nr_features() ) )
        support_vector_labels = sparse.lil_matrix( ( nr_cvs, 1 ) )

        for i in range( nr_cvs ):
            support_vectors[i, :] = self.dataset.get_point( self.coreset[i] )
            support_vector_labels[i, 0] = self.labels[self.coreset[i]]
            alphas[i, 0] = self.alphas[i, 0]

        support_vectors = support_vectors.tocsr()
        support_vector_labels = support_vector_labels.tocsr()
        alphas = alphas.tocsr()

        model = common.model.Model( alphas, self.__get_b(), self.class_name,
                                    nr_cvs, support_vector_labels,
                                    support_vectors )
        return model

    def __cvm_kernel ( self, i, j ):
        ''' The CVM kernel function

            TODO ( if slow ): Implement caching
        '''
        d_ij = 0.0
        if i == j:
            d_ij = 1.0

        t_ij = self.labels[i] * self.labels[j]
        return  t_ij * self.__kernel( i, j ) + t_ij + ( d_ij / self.C )

    def __kernel( self, i, j ):
        ''' 
        The actual kernel function. It is a normalised dot product for now
            
        TODO ( someday ): Implement other kernels as well
        TODO ( if slow ): Implement a caching scheme
        '''
        k_ij = self.dataset.get_sparse_point( i ) * self.dataset.get_sparse_point( j ).T
        k_ii = self.dataset.get_sparse_point( i ) * self.dataset.get_sparse_point( i ).T
        k_jj = self.dataset.get_sparse_point( j ) * self.dataset.get_sparse_point( j ).T

        return ( k_ij[0, 0] / ( np.math.sqrt( k_ii[0, 0] ) * np.math.sqrt( k_jj[0, 0] ) ) )

    def __get_b( self ):
        ''' Calculate the current bias '''
        b = 0.0
        n = len( self.coreset )

        for i in range( n ):
            b += self.alphas[i, 0] * self.labels[self.coreset[i]]

        return b

    def __get_ro( self ):
        ''' Calculate the current Ro '''
        return np.dot( np.dot( self.alphas.T, self.k_cache ), self.alphas )[0, 0]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
