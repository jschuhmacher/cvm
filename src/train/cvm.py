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

# Local
import common.data
import common.model
import common.kernels as kernels
import classify.predictor

random.seed( 400 )


''' Size of the initial set of samples taken '''
INITIAL_CS = 2

''' Size of the initial epsilon '''
INITIAL_EPS = 0.1
EPS_SCALE = 0.5

''' Size of the set of samples taken at each iteration '''
NR_SAMPLES = 60

''' Maximum number of iterations to train '''
MAX_ITERS = 150

''' Maximum number of iterations to train '''
MAX_TRIES = 7

class CVM( object ):
    ''' 
    Implementation of the Core Vector Machine described in the 2006 paper 
    by Tsang et al. "Core Vector Machines: Fast SVM Training on Very Large 
    Data Sets" published in The Journal of Machine Learning Research.
    '''

    # Frustrating fact: Python class attributes are global to all
    # instances of that class

    def __init__( self, class_name, dataset, EPSILON=1.0e-6, kernel=0,
                  C=1e6, gamma=0.01 ):
        self.__NR_SAMPLES = NR_SAMPLES

        ''' Acceptable approximation for the MEB '''
        self.__EPSILON = EPSILON

        ''' Regularisation constant '''
        self.__C = C

        ''' RBF kernel '''
        self.__gamma = gamma

        ''' The dataset used for training. '''
        self.__dataset = dataset

        ''' The class this classifier will learn. '''
        self.__class_name = class_name

        if kernel == 0:
            self.__kernel = kernels.norm_dot_kernel
        elif kernel == 1:
            self.__kernel = kernels.rbf_kernel
        else:
            logging.fatal( '  Unkown kernel type' )

        ''' 
        The core set: S
        A list of pointers into the dataset
        '''
        self.__coreset = []

        ''' The squared radius of the ball: R^2 '''
        self.__radius2 = 0.0

        ''' Class labels, translated to {-1, +1}, for the current class '''
        self.__labels = []

        ''' List of pointers to all positive records. '''
        self.__pos_ids = []

        ''' List of pointers to all negative records. '''
        self.__neg_ids = []

        ''' Data distribution for this class '''
        self.__nr_pos = 0
        self.__nr_neg = 0

        ''' Should next sample be positive or negative? '''
        self.__pos_turn = True

        ''' Lagrange multipliers '''
        self.__alphas = np.zeros( ( 1, 1 ), np.float )

        ''' Small cache for the core vector kernel evaluations '''
        self.__k_cache = np.zeros( ( 2, 2 ), np.float )

        self.__iteration = 0

        # Solver options
        cvxopt.solvers.options['show_progress'] = False

        # Maximum number of iterations 
        cvxopt.solvers.options['maxiters'] = 100

        # Absolute accuracy
        cvxopt.solvers.options['abstol'] = self.__EPSILON * self.__EPSILON

        # Relative accuracy
        cvxopt.solvers.options['reltol'] = self.__EPSILON * self.__EPSILON

        # Tolerance for feasibility condition
        cvxopt.solvers.options['feastol'] = 1e-12
        cvxopt.solvers.options['refinement'] = 2

        logging.debug( ( "    Classifier for <{0}> starts with EPSILON: {1}, C:" +
                       " {2}, NR_SAMPLES: {3}" ).format( self.__class_name,
                               self.__EPSILON,
                               self.__C,
                               self.__NR_SAMPLES ) )

        # Count class distribution
        for i in range( self.__dataset.get_nr_points() ):
            label = self.__dataset.get_label( i )
            if label == self.__class_name:
                self.__labels.append( 1 )
                self.__pos_ids.append( i )
                self.__nr_pos += 1
            else:
                self.__labels.append( -1 )
                self.__neg_ids.append( i )
                self.__nr_neg += 1

        logging.debug( "    Class distribution for class <{0}>:{1}, <other>:{2}.".
                       format( self.__class_name, self.__nr_pos, self.__nr_neg ) )

    def train ( self ):
        ''' 
        Learns a linear model from training examples with 2 classes.
        '''
        self.__initialise()
        self.__iteration = 0

        tries = 0
        currentEpsilon = INITIAL_EPS
        while( tries < MAX_TRIES and self.__iteration < MAX_ITERS ):

            if currentEpsilon < self.__EPSILON:
                currentEpsilon = self.__EPSILON

            eps_factor = ( 1. + currentEpsilon )
            eps_factor *= eps_factor

            new_cs = self.__furthest_point( eps_factor * self.__radius2 )

            # No more points remain
            if new_cs == None and tries >= MAX_TRIES:
                logging.info( "  Completed training for class <{0}> on max tries.".
                              format( self.__class_name ) )
                break
            elif new_cs == None:
                tries += 1
            else:
                tries = 0
                self.__update_coreset( new_cs )
                self.__solve_qp()
                self.__update_radius2()
                self.__iteration += 1

        logging.info( "  Completed training for class <{0}> in {1} iterations".
                      format( self.__class_name, self.__iteration ) )

        return self.__make_model()

    def __initialise( self ):
        ''' 
        Initialise S_0, c_0 and R_0
        '''

        # Initialize the core set with a balanced sample
        for i in range( INITIAL_CS ):
            z_i = self.__next_sample()
            if z_i != None:
                self.__update_coreset( z_i )
                self.__alphas[i, 0] = ( 1.0 / INITIAL_CS )

        # Construct a cache of kernel evaluations of the core vectors
        n = len( self.__coreset )
        for i in range( n ):
            for j in range( n ):
                self.__k_cache[i, j] = self.__cvm_kernel( i, j )

        self.__solve_qp()

        # Compute a new radius
        self.__update_radius2()

    def __next_sample( self ):
        ''' 
        Returns an id in the list of datapoints. Tries to return balanced
        random samples when enough data is present.
        '''
        # No points left
        if self.__nr_pos <= 0 and self.__nr_neg <= 0:
            return None

        if self.__nr_pos <= 0:
            self.__pos_turn = False
        elif self.__nr_neg <= 0:
            self.__pos_turn = True
        else:
            self.__pos_turn = not self.__pos_turn

        # Select a random point that has not yet been used
        # if self.__pos_turn:
        #r = random.randint( 1, 10 )
        #if r <= 5 and self.__nr_pos > 0:
        #    id = self.__pos_ids[random.randint( 0, self.__nr_pos - 1 )]
        #elif r > 5 and self.__nr_neg > 0:
        #    id = self.__neg_ids[random.randint( 0, self.__nr_neg - 1 )]
        if self.__pos_turn:
            id = self.__pos_ids[random.randint( 0, self.__nr_pos - 1 )]
        else:
            id = self.__neg_ids[random.randint( 0, self.__nr_neg - 1 )]

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
        
            minimise: 0.5 \alpha^T (__cvm_kernel) \alpha + 0^T \alpha
             \alpha
            
            s.t.: -1^T \alpha  <= 0
                   1^T \alpha   = 1
        
        to make it compatible with cvxopt.
        '''
        n = len( self.__coreset )

        P = cvxopt.matrix( self.__k_cache )

        q = cvxopt.matrix( 0.0, ( n, 1 ) )
        G = cvxopt.matrix( 0.0, ( n, n ) )
        G[:: n + 1] = -1.0
        h = cvxopt.matrix( 0.0, ( n, 1 ) )
        A = cvxopt.matrix( 1.0, ( 1, n ) )
        b = cvxopt.matrix( 1.0 )
        x = cvxopt.matrix( self.__alphas )

        logging.info( "  {0} Solving QP for iteration {1}".format( self.__class_name,
                                                                   self.__iteration ) )

        solution = cvxopt.solvers.qp( P, q, G, h, A, b, None, {'x': x} )

        if solution['status'] != 'optimal':
            logging.debug( '    {0} Could not solve QP problem, results: {1}\n'.
                           format( self.__class_name, pprint.pformat( solution ) ) )

        # Update multipliers
        self.__alphas = np.asarray( solution['x'] ).copy()
        print self.__class_name, self.__iteration, pprint.pformat(solution)

        #logging.debug( '  New Lagrange multipliers:\n' + pprint.pformat( self.__alphas ) )

    def __update_radius2( self ):
        ''' 
        Update the MEB.
            
        Calculate the new radius:
            R^2 = \alpha^T diag(K) - \alpha^T K \alpha
        
        as \alpha_i is 0 for non CVs, only entries corresponding to CVs need to 
        be evaluated.
        '''
        self.__radius2 = np.dot( self.__alphas.T, self.__k_cache.diagonal() )[0] - \
                                 self.__get_ro()

        logging.info( "  {0} New squared radius: {1:<10.11}".format( self.__class_name,
                                                                     self.__radius2 ) )

        # The radius is squared, it should be positive
        assert( self.__radius2 >= 0.0 )

    def __distance_to_centroid( self, z_l ):
        ''' 
        Computes the squared distance from a point to the center of
        the coreset: 

            ||c_t - \phi(point)||^2 
            = \sum_{i,j} \alpha_i \alpha_j __cvm_kernel(z_i, z_j) 
            - 2 \sum_i \alpha_i __cvm_kernel(z_i, z_l) __cvm_kernel(z_l, z_l)
        
        see Eq. 18 in the paper.
        '''
        cs_size = len( self.__coreset )

        distance2 = 0.0
        for i in range( cs_size ):
            distance2 += self.__alphas[i, 0] * \
                         self.__cvm_kernel( self.__coreset[i], z_l )

        return ( self.__get_ro() - 2.0 * distance2 + self.__cvm_kernel( z_l, z_l ) )

    def __furthest_point( self, prev_max_distance ):
        ''' 
        Take NR_SAMPLES samples and select the one point that is furthest away 
        from the current center c_t.
        '''
        max_distance_id = None
        max_distance = prev_max_distance

        for i in range( NR_SAMPLES ):
            new_id = self.__next_sample()

            # No more points 
            if new_id == None:
                break

            new_distance = self.__distance_to_centroid( new_id )
            if new_distance >= max_distance:
                max_distance = new_distance
                max_distance_id = new_id

        return max_distance_id

    def __update_coreset( self, max_dist_id ):
        ''' Add new points to the coreset '''
        assert( max_dist_id != None )
        assert( ( len( self.__pos_ids ) +
                  len( self.__neg_ids ) ) <= len( self.__labels ) )

        if self.__labels[max_dist_id] == 1:
            self.__nr_pos -= 1
            self.__pos_ids.remove( max_dist_id )
        elif self.__labels[max_dist_id] == -1:
            self.__nr_neg -= 1
            self.__neg_ids.remove( max_dist_id )
        else:
            assert( self.__labels[max_dist_id] == 1 or
                    self.__labels[max_dist_id] == -1 )

        self.__coreset.append( max_dist_id )
        self.__alphas.resize( ( len( self.__coreset ), 1 ) )

        # Only do this for the newly added CV, the rest of the matrix can be 
        # copied from the previous iteration.
        n = len( self.__coreset )
        if n > 2:
            new_row = np.zeros( ( 1, n ), np.float )
            new_column = np.zeros( ( n - 1, 1 ), np.float )

            for i in range( n ):
                if i < ( n - 1 ):
                    new_column[i, 0] = self.__cvm_kernel( i, n - 1 )
                new_row[0, i] = self.__cvm_kernel( n - 1, i )

            self.__k_cache = np.hstack( [self.__k_cache, new_column] )
            self.__k_cache = np.vstack( [self.__k_cache, new_row] )
        else:
            for i in range( n ):
                for j in range( n ):
                    self.__k_cache[i, j] = self.__cvm_kernel( i, j )

    def __make_model( self ):
        ''' 
        Store the lagrange multipliers, bias and core vectors. These will
        be used for classifying unseen points. 
        '''
        logging.info( '  Creating model' )
        nr_cvs = len( self.__coreset )
        alphas = sparse.lil_matrix( ( nr_cvs, 1 ) )
        support_vectors = []
        support_vector_labels = sparse.lil_matrix( ( nr_cvs, 1 ) )

        for i, cs in enumerate( self.__coreset ):
            support_vectors.append( self.__dataset.get_point( cs ) )
            support_vector_labels[i, 0] = self.__labels[cs]
            alphas[i, 0] = self.__alphas[i, 0]

        # Storage in scipy sparse matrix format
        support_vectors = np.asarray( support_vectors )

        support_vectors = sparse.csr_matrix( support_vectors, dtype=np.float )
        support_vector_labels = support_vector_labels.tocsr()
        alphas = alphas.tocsr()

        model = common.model.Model( alphas, self.__get_b(), self.__class_name,
                                    nr_cvs, support_vector_labels,
                                    support_vectors, self.__kernel, self.__gamma,
                                    self.__get_ro(), self.__C )
        return model


    def __cvm_kernel ( self, i, j ):
        ''' 
        The CVM kernel function. A wrapper around a module compiled with 
        Cython.
        '''
        return kernels.cvm_kernel( self.__labels[i],
                                   self.__labels[j],
                                   self.__dataset.get_sparse_point( i ),
                                   self.__dataset.get_sparse_point( j ),
                                   i, j, self.__kernel, self.__gamma, self.__C )

    def __get_b( self ):
        ''' Calculate the current bias '''
        b = 0.0
        n = len( self.__coreset )

        for i in range( n ):
            b += ( self.__alphas[i, 0] * self.__labels[self.__coreset[i]] )

        return b

    def __get_ro( self ):
        ''' Calculate the current Ro '''
        return np.dot( np.dot( self.__alphas.T, self.__k_cache ),
                       self.__alphas )[0, 0]

if __name__ == '__main__':
    import doctest
    doctest.testmod()
