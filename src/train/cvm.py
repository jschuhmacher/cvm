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
import scipy.sparse as spsparse
import cvxopt
import cvxopt.solvers

# Local
import common.data
import common.model
import classify.predictor

random.seed( 200 )

class CVM( object ):
    ''' 
    Implementation of the Core Vector Machine described in the 2006 paper 
    by Tsang et al. "Core Vector Machines: Fast SVM Training on Very Large 
    Data Sets" published in The Journal of Machine Learning Research.
    '''

    #---------------------------------------------------------------------------
    # CVM parameters
    #---------------------------------------------------------------------------
    ''' Size of the set of samples taken at each iteration '''
    NR_SAMPLES = None

    ''' Acceptable approximation for the MEB '''
    EPSILON = None

    ''' 'Rgularisation constant '''
    C = None

    #---------------------------------------------------------------------------
    ''' The dataset used for training. '''
    dataset = None

    ''' The class this classifier will learn. '''
    class_name = None

    #---------------------------------------------------------------------------
    ''' The core set: S
        A list of pointers into the dataset
    '''
    coreset = []

    ''' The squared radius of the ball: R^2 '''
    radius2 = None

    ''' Class labels, translated to {-1, +1}, for the current class '''
    labels = []

    ''' List of pointers to all positive records. '''
    pos_ids = []

    ''' List of pointers to all negative records. '''
    neg_ids = []

    ''' Data distribution for this class '''
    nr_pos = 0
    nr_neg = 0

    ''' Should next sample be positive or negative? '''
    pos_turn = True

    ''' Lagrange multipliers '''
    alphas = {}
    alphas_cvx = cvxopt.matrix( 1.0 )

    ''' Small cache for the core vector kernel evaluations '''
    k_cache = cvxopt.matrix( 1.0 )

    #---------------------------------------------------------------------------

    def __init__( self, class_name, dataset, C=1000.0, EPSILON=1.0e-6, NR_SAMPLES=20 ):
        ''' '''
        self.dataset = dataset
        self.class_name = class_name
        self.NR_SAMPLES = NR_SAMPLES
        self.EPSILON = EPSILON
        self.C = C

        # Solver options
        cvxopt.solvers.options['show_progress'] = False

        # Maximum number of iterations 
        cvxopt.solvers.options['maxiters'] = 100

        # Absolute accuracy
        cvxopt.solvers.options['abstol'] = 1e-20

        # Relative accuracy
        cvxopt.solvers.options['reltol'] = 0

        # Tolerance for feasibility condition
        cvxopt.solvers.options['feastol'] = 1e-20
        cvxopt.solvers.options['refinement'] = 0

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
        #if self.nr_pos > self.NR_SAMPLES:
        self.__initialise()
        iteration = 0

#            while( self.__has_remaining_point() ):
        while( True ):
            new_cs = self.__furthest_point( self.radius2 + self.EPSILON )

            # No more points remain
            if new_cs == None:
                break
            else:
                self.__update_coreset( new_cs )
                self.__solve_qp()
                self.__update_radius2()
                iteration += 1

            if iteration % 20 == 0:
                logging.debug( "  Iteration {0}".format( iteration ) )

        logging.info( "  Completed training for class <{0}> in {1} iterations".
                      format( self.class_name, iteration ) )
#        else:
#            logging.info( "  Skipping training for class <{0}>: too little data".
#                          format( self.class_name ) )

        return self.__make_model()

    def __make_model( self ):
        ''' Store results in the model '''

        logging.info( '  Creating model' )
        alphas = []
        support_vectors = []
        support_vector_labels = []

        for i in self.coreset:
            support_vectors.append( self.dataset.get_point( i ) )
            support_vector_labels.append( self.labels[i] )
            alphas.append( self.alphas[i] )

        support_vectors = np.asarray( support_vectors, np.float32 )
        support_vector_labels = np.asarray( support_vector_labels, np.float32 )
        alphas = np.asarray( alphas, np.float32 )
        nr_svs = len( support_vectors )

        model = common.model.Model( alphas, self.__get_b(), self.class_name,
                                    nr_svs, support_vector_labels,
                                    support_vectors )
        return model


    def __initialise( self ):
        ''' 
        Initialise S_0, c_0 and R_0
        '''
        # Initialise the coreset with a balanced sample
        z_a = self.__next_sample()
        self.__update_coreset( z_a )
        self.alphas[z_a] = 0.5

        z_b = self.__next_sample()
        self.__update_coreset( z_b )
        self.alphas[z_b] = 0.5

        self.__update_alphas()

        # Construct a cache of kernel evaluations of the core vectors
        n = len( self.coreset )
        self.k_cache = cvxopt.matrix( 0.0, ( n, n ) )
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

        assert( self.nr_pos > 0 or self.nr_neg > 0 )

        if self.nr_pos <= 0:
            self.pos_turn = False
        elif self.nr_neg <= 0:
            self.pos_turn = True
        else:
            self.pos_turn = not self.pos_turn

        id = None
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

        #temp = self.k_cache.copy()
        temp = self.k_cache.copy

        self.k_cache = cvxopt.matrix( 0.0, ( n, n ) )

        # Fill P[i, j] with kernel evaluations for i and j in the core set
        # Only do this for the newly added CV, the rest of the matrix can be 
        # copied from the previous iteration. 
        for i in range( n ):
            for j in range( n ):
                if j == n - 1 or i == n - 1:
                    self.k_cache[i, j] = self.__cvm_kernel( i, j )
                else:
                    self.k_cache[i, j] = temp[i, j]

        logging.debug( pprint.pformat( self.k_cache ) )
        q = cvxopt.matrix( 0.0, ( n, 1 ) )

        G = cvxopt.matrix( 0.0, ( n, n ) )
        G[::n + 1] = -1.0
        h = cvxopt.matrix( 0.0, ( n, 1 ) )
        A = cvxopt.matrix( 1.0, ( 1, n ) )
        b = cvxopt.matrix( 1.0 )

        # Copy multipliers to cvxopt matrix format to provide a warm start
        self.__update_alphas()

        logging.debug( "  Solving QP {0}".format( iteration ) )
        solution = cvxopt.solvers.qp( 2.0 * self.k_cache, q, G, h, A, b, None,
                                      {'x': self.alphas_cvx} )

        logging.debug( pprint.pformat( solution ) )

        # Update multipliers
        self.alphas_cvx = solution['x']
        for i in range( n ):
            self.alphas[self.coreset[i]] = solution['x'][i, 0]

        logging.debug( pprint.pformat( self.alphas ) )

    def __update_alphas( self ):
        ''' Copy the dict of alphas to a faster data structure '''
        n = len( self.coreset )
        self.alphas_cvx = cvxopt.matrix( 0.0, ( n, 1 ) )
        for i in range( n ):
            self.alphas_cvx[i, 0] = self.alphas[self.coreset[i]]

    def __update_radius2( self ):
        ''' 
        Update the MEB.
            
        Calculate the new radius:
            R^2 = \alpha^T diag(K) - \alpha^T K \alpha
        
        as \alpha_i is 0 for non CVs, only entries corresponding to CVs need to 
        be evaluated.
        '''
        n = len( self.coreset )

        # TODO: this can probably be done much faster
        diag_K = cvxopt.matrix( 0.0, ( n, 1 ) )
        for i in range( n ):
            diag_K[i, 0] = self.k_cache[i, i]

        r1 = ( self.alphas_cvx.T * diag_K )[0]
        r2 = ( self.alphas_cvx.T * self.k_cache * self.alphas_cvx )[0]
        print r1
        print r2
        stop
        #self.radius2 = abs( r1 - r2 )
        logging.debug( "Radius^2: {0} - {1} = {2}".format( r1, r2, self.radius2 ) )

        # The radius is squared, it should be positive
        # assert( self.radius2 >= 0.0 )

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
        distance1 = 0.0
        distance2 = 0.0

        for i in range( cs_size ):
            for j in range( cs_size ):
                distance1 += self.alphas[self.coreset[i]] * \
                             self.alphas[self.coreset[j]] * \
                             self.__cvm_kernel( self.coreset[i], self.coreset[j] )

            distance2 += self.alphas[self.coreset[i]] * \
                         self.__cvm_kernel( self.coreset[i], z_l ) + \
                         self.__cvm_kernel( z_l, z_l )

        return ( distance1 - 2.0 * distance2 )

    def __furthest_point( self, prev_max_distance ):
        ''' 
        Take NR_SAMPLES samples and select the one point that is furthest away 
        from the current center c_t.
        '''
        max_distance_id = None
        max_distance = prev_max_distance

        for i in range( self.NR_SAMPLES ):
            new_id = self.__next_sample()
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
        #assert( ( len( self.pos_ids ) + len( self.neg_ids ) ) == len( self.labels ) )
        #print len( self.pos_ids )
        #print len( self.neg_ids )
        #print len( self.labels )

        #print len( self.pos_ids )
        #print len( set( self.pos_ids ) )
        #stop
        print max_dist_id
        print self.pos_ids
        print self.neg_ids
        print self.labels

        try:
            if self.labels[max_dist_id] == 1:
                self.nr_pos -= 1
                self.pos_ids.remove( max_dist_id )
            elif self.labels[max_dist_id] == -1:
                self.nr_neg -= 1
                self.neg_ids.remove( max_dist_id )
        except:
            logging.error( '  Not found {0}'.format( max_dist_id ) )
            pass

        self.coreset.append( max_dist_id )
        self.alphas[max_dist_id] = 0.0
        self.__update_alphas()

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
        ''' The actual kernel function, only dot product for now

            TODO ( someday ): Implement other kernels as well
            TODO ( if slow ): Implement a caching scheme
        '''
        return np.inner( self.dataset.get_point( i ),
                         self.dataset.get_point( j ) )

    def __get_b( self ):
        ''' Calculate the current bias '''
        b = 0.0
        n = len( self.coreset )

        for i in range( n ):
            b += self.alphas_cvx[i] * self.labels[self.coreset[i]]

        return b

    def __get_ro( self ):
        ''' Calculate the current Ro '''
        ro = 0.0
        n = len( self.coreset )

        for i in range( n ):
            for j in range( n ):
                ro += self.alphas_cvx[i] * self.alphas_cvx[j] * \
                      self.k_cache[i, j]

        return ro


if __name__ == '__main__':
    import doctest
    doctest.testmod()
