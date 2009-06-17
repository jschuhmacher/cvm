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

This is the "data" module. It contains data structures for labelled and unlabel-
led sparse data sets, for use within a linear classifier. It provides several 
convenience methods, e.g. for IO, splitting, shuffling, normalisation etc. 
'''

import logging
import sys
import random
import numpy as np
import scipy as sp
import pprint

random.seed( 200 )

class Dataset( object ):
    ''' Data structure representing a dataset.
    '''

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

    ''' A list of classes '''
    __classes = None

    ''' "Feature: type" pairs. Type is either continuous or symbolic. '''
    __features = None

    ''' Number of symbolic features '''
    __nr_symbolic = 0

    ''' Number of continuous features '''
    __nr_continuous = 0

    ''' List of lists of features '''
    __data = None

    ''' Class labels, if present '''
    __labels = None

    ''' Is this dataset labelled? '''
    __has_labels = False

    ''' Number of points in the dataset '''
    __nr_points = 0

    ''' List of pointers into the dataset '''
    __indices = []

    def read ( self, header_file, data_file ):
        ''' Loads the data '''
        self.__read_header( header_file )
        self.__read_data( data_file )

        # Create indices to the dataset
        self.__indices = [x for x in range( self.__nr_points )]

    def shuffle ( self ):
        ''' Shuffles the data in this dataset '''
        logging.info( "  Shuffling dataset" )
       # random.shuffle( self.__indices )

    def split( self, percentage=66.6 ):
        ''' Returns 2 subsets of this dataset according to the parameter
        '''
        logging.info( '    TODO: split' )
        pass

    def preprocess ( self ):
        ''' 
        Preprocesses the values in this dataset.
            
        First removes all symbolic features by converting them to indicator
        features. For every possible value a symbolic feature has in the 
        data, an indicator will be generated.  
        '''
        continuous_data = []

        logging.debug( '  Converting to continuous data' )
        for connection in self.__data:
            continuous_connection = []

            for i in range( self.__nr_symbolic + self.__nr_continuous ):
                # Symbolic features are converted to indicators
                if self.__features[1][i] == 'symbolic':
                    for j in self.__features[2]:
                        if j == connection[i]:
                            continuous_connection.append( 1.0 )
                        else:
                            continuous_connection.append( 0.0 )
                # Continuous features are copied
                elif self.__features[1][i] == 'continuous':
                    continuous_connection.append( connection[i] )
                else:
                    raise ValueError( "Undefined feature type" )

            continuous_data.append( continuous_connection )

        logging.debug( '  Converting to numpy format' )
        self.__data = np.asarray( continuous_data, np.float32 )
        continuous_data = None
        #self.__data = sp.sparse.csr_matrix( self.__data )

    def normalise ( self ):
        ''' Normalises the values in this dataset to the range 0..1 '''
        maxima = np.amax( self.__data, axis=0 )

        empty = []
        for i in range( maxima.size ):
            if maxima[i] == 0.0:
                # Empty column
                empty.append( 0 )
            else:
                empty.append( 1 )

        logging.debug( '  Pruning empty features' )
        # Prune dataset and maxima
        self.__data = np.compress( empty, self.__data, axis=1 )
        maxima = np.compress( empty, maxima, axis=0 )

        logging.debug( '  {0} out of {1} features left'.format( maxima.size, len( empty ) ) )

        logging.info( '  Normalising data' )
        # Scale dataset between 0 and 1 
        self.__data = self.__data / maxima

    def get_indices( self ):
        ''' Returns indices into this dataset '''
        return self.__indices

    def get_point( self, index ):
        ''' Returns the data point at index '''
        return self.__data[index]

    def get_label( self, index ):
        ''' Returns the label for the point at index '''
        return self.__labels[index]

    def get_classes ( self ):
        ''' Returs a list of classes '''
        return self.__classes

    def __read_header( self, header_file ):
        ''' Read the "header"-file <dataset-name.names> '''

        # Classes are in the header file
        self.__classes = header_file.readline().strip( '.\n' ).split( ',' )
        # But for now they are hard coded
        self.__classes = ['normal', 'u2r', 'dos', 'r2l', 'probe']

        self.__features = ( [], [], [] )
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
                self.__register_symbols( connection[:-1] )
            else:
                self.__data.append( connection )
                self.__register_symbols( connection )
            self.__nr_points += 1

        logging.info( '  Read {0} connections from file'.
                      format( self.__nr_points ) )

        # Convert list of sets in self.__features to a list of lists, to 
        # fixate the ordering.
        self.__features = self.__features[0], self.__features[1], map ( list, self.__features[2] )


        if self.__has_labels:
            logging.info( '    labels are present' )
        else:
            logging.info( '    labels are not present' )

    def __register_symbols( self, connection ):
        ''' Register the various occurances of feature-values for symbolic 
            features 
        '''
        for i in range( self.__nr_continuous + self.__nr_symbolic ):
            if self.__features[1][i] == 'symbolic':
                self.__features[2][i].add( connection[i] )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
