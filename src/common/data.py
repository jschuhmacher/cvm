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
File: data.py

This is the "data" module. It contains data structures for labelled and 
unlabelled sparse data sets, for use within a linear classifier. It should 
eventually provide several convenience methods, e.g. for IO, splitting, 
shuffling, etc. 
'''

# System
import logging
import sys
import random
import pprint
import cPickle as pickle

# 3rd-party
import numpy as np
from scipy import sparse


# Fix random seed
random.seed( 200 )

class Dataset( object ):
    ''' Data structure representing a dataset.
    '''

    # To be more useful with other datasets eventually this should be removed, 
    # either the dataset itself will need to be changed, or some more general 
    # mechanism will need to be implemented.
    TRAINING_ATTACK_TYPES = {
        'normal': 'normal',
        'back': 'dos',
        'buffer_overflow': 'u2r',
        'ftp_write': 'r2l',
        'guess_passwd': 'r2l',
        'imap': 'r2l',
        'ipsweep': 'probe',
        'land': 'dos',
        'loadmodule': 'u2r',
        'multihop': 'r2l',
        'neptune': 'dos',
        'nmap': 'probe',
        'perl': 'u2r',
        'phf': 'r2l',
        'pod': 'dos',
        'portsweep': 'probe',
        'rootkit': 'u2r',
        'satan': 'probe',
        'smurf': 'dos',
        'spy': 'r2l',
        'teardrop': 'dos',
        'warezclient': 'r2l',
        'warezmaster': 'r2l',
    }

    def __init__( self ):
        ''' A list of classes '''
        self.__classes = None

        ''' "Feature: type" pairs. Type is either continuous or symbolic. '''
        self.__features = None

        ''' Number of symbolic features '''
        self.__nr_symbolic = 0

        ''' Number of continuous features '''
        self.__nr_continuous = 0

        ''' List of lists of features '''
        self.__data = None

        ''' Class labels, if present '''
        self.__labels = None

        ''' Is this dataset labelled? '''
        self.__has_labels = False

        ''' Number of points in the dataset '''
        self.__nr_points = 0

        ''' List of pointers into the dataset '''
        self.__indices = []

    def read ( self, header_file, data_file ):
        ''' Loads the data '''

        self.__read_header( header_file )

        # XXX: Overwrite the features with a list derived from the entire dataset
        # HACK
        self.__features = pickle.load( open( header_file.name + '_total', 'r' ) )

        pickled = False
        try:
            cache_file = open( data_file.name + '_cache', 'r' )
            self.__data = pickle.load( cache_file )
            self.__labels = pickle.load( cache_file )
            self.__nr_points = self.__data.shape[0]
            pickled = True
        except Exception:
            pass

        if not pickled:
            self.__read_data( data_file )
            self.preprocess()
            cache_file = open( data_file.name + '_cache', 'w' )
            pickle.dump( self.__data, cache_file, pickle.HIGHEST_PROTOCOL )
            pickle.dump( self.__labels, cache_file, pickle.HIGHEST_PROTOCOL )

        self.__indices = [x for x in range( self.__nr_points )]

    def shuffle ( self ):
        ''' Shuffles the data in this dataset '''
        logging.info( "  Shuffling dataset" )
        random.shuffle( self.__indices )

    def split( self, percentage=66.6 ):
        ''' Returns 2 subsets of this dataset according to the parameter
        '''
        logging.info( '    TODO: split' )
        pass

    def get_nr_features( self ):
        return ( sum( map( len, self.__features[2] ) ) + self.__nr_continuous )

    def preprocess ( self ):
        ''' 
        Preprocesses the values in this dataset.
            
        First removes all symbolic features by converting them to indicator
        features. For every possible value a symbolic feature has in the 
        data, an indicator will be generated.  
        '''
        total_nr_features = sum( map( len, self.__features[2] ) ) + \
                            self.__nr_continuous

        # Create a sparse matrix of nr_points rows and nr_features columns
        continuous_data = sparse.dok_matrix( ( self.__nr_points, total_nr_features ), dtype=np.float )

        logging.debug( '  Converting to continuous data' )
        for i in range( self.__nr_points ):
            for j in range( self.__nr_symbolic + self.__nr_continuous ):
                # Symbolic features are converted to indicators
                if self.__features[1][j] == 'symbolic':
                    for symbol in self.__features[2][j]:
                        if self.__data[i][j] == symbol:
                            continuous_data[i, j] = 1.0
#                        else:
#                            continuous_data[i, j] = 0.0
                # Continuous features are copied
                elif self.__features[1][j] == 'continuous':
                    if self.__data[i][j] != 0:
                        continuous_data[i, j] = self.__data[i][j]
                else:
                    raise ValueError( "Undefined feature type: {0}".format( symbol ) )

        self.__data = continuous_data.tocsr()

    def get_indices( self ):
        ''' Returns indices into this dataset '''
        return self.__indices

    def get_point( self, index ):
        ''' Returns the data point at index '''
        return self.__data[index, :].todense()

    def get_sparse_point( self, index ):
        ''' Returns the data point at index '''
        return self.__data[index]

    def get_label( self, index ):
        ''' Returns the label for the point at index '''
        return self.__labels[index]

    def get_classes ( self ):
        ''' Returns a list of classes '''
        return self.__classes

    def __read_header( self, header_file ):
        ''' Read the "header"-file <dataset-name.names> '''

        # Classes are in the header file
        self.__classes = header_file.readline().strip( '.\n' ).split( ',' )
        # But for now they are hard coded
        self.__classes = ['normal', 'u2r', 'dos', 'r2l', 'probe']

        self.__features = ( [], # Feature names
                            [], # Feature type: continuous or symbolic
                            [], # If symbolic: a set of possible symbols
                                # otherwise: empty set 
                          )
        while True:
            feature = header_file.readline()
            if len( feature ) == 0:
                break

            feature = feature.strip( '.\n' ).split( ': ' )
            if feature[1] == 'continuous':
                self.__nr_continuous += 1
            elif feature[1] == 'symbolic':
                self.__nr_symbolic += 1
            else:
                raise ValueError( "Unknown feature type encountered: [{0}]".
                                  format( feature[1] ) )

            self.__features[0].append( feature[0] )
            self.__features[1].append( feature[1] )
            self.__features[2].append( set() )

        logging.info( '  Read {0} features from file'.
                      format( self.__nr_continuous + self.__nr_symbolic ) )
        logging.info( '    of which {0} are symbolic and {1} are continuous'.
                      format( self.__nr_symbolic, self.__nr_continuous ) )

    def __read_data( self, data_file ):
        ''' Read the actual data '''
        self.__data = list()
        self.__labels = list()
        for connection in data_file:
            connection = connection.rstrip( '.\n' )
            connection = connection.split( ',' )

            if self.__has_labels or len( connection ) == ( self.__nr_continuous +
                                                           self.__nr_symbolic + 1 ):
                self.__has_labels = True
                self.__labels.append( self.TRAINING_ATTACK_TYPES[
                     connection[self.__nr_continuous + self.__nr_symbolic]] )
                self.__data.append( connection[:-1] )
                # self.__register_symbols( connection[:-1] )
            else:
                self.__data.append( connection )
                # self.__register_symbols( connection )
            self.__nr_points += 1

        logging.info( '  Read {0} connections from file'.
                      format( self.__nr_points ) )

        # Convert list of sets in self.__features to a list of lists, to 
        # fixate the ordering.
        # self.__features = self.__features[0], self.__features[1], ( map ( list, self.__features[2] ) )
        # pprint.pprint( self.__features, sys.stdout )

        if self.__has_labels:
            logging.info( '    labels are present' )
        else:
            logging.info( '    labels are not present' )

    def __register_symbols( self, connection ):
        ''' 
        Register the various occurrences of feature-values for symbolic 
        features.
        '''
        for i in range( self.__nr_continuous + self.__nr_symbolic ):
            if self.__features[1][i] == 'symbolic':
                self.__features[2][i].add( connection[i] )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
