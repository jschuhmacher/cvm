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
import pprint
import cython

# 3rd-party
import numpy as np
from scipy import sparse

class Dataset( object ):
    ''' Data structure representing a dataset.
    '''

    # To be more useful with other datasets eventually this should be removed, 
    # either the dataset itself will need to be changed, or some more general 
    # mechanism will need to be implemented.
    TRAINING_ATTACK_TYPES = {
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
        'normal': 'normal',
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
        'apache2':'dos',
        'back':'dos',
        'buffer_overflow':'u2r',
        'ftp_write':'r2l',
        'guess_passwd':'r2l',
        'httptunnel':'r2l',
        'httptunnel':'u2r',
        'imap':'r2l',
        'ipsweep':'probe',
        'land':'dos',
        'loadmodule':'u2r',
        'mailbomb':'dos',
        'mscan':'probe',
        'multihop':'r2l',
        'multihop':'u2r', # note that this is a duplicate
        'named':'r2l',
        'neptune':'dos',
        'nmap':'probe',
        'perl':'u2r',
        'phf':'r2l',
        'pod':'dos',
        'portsweep':'probe',
        'processtable':'dos',
        'ps':'u2r',
        'rootkit':'u2r',
        'saint':'probe',
        'satan':'probe',
        'sendmail':'r2l',
        'smurf':'dos',
        'snmpgetattack':'r2l',
        'snmpguess':'r2l',
        'sqlattack':'u2r',
        'teardrop':'dos',
        'udpstorm':'dos',
        'warezmaster':'dos',
        'worm':'r2l',
        'xlock':'r2l',
        'xsnoop':'r2l',
        'xterm':'u2r',
    }

    #    TRAINING_ATTACK_TYPES = {
    #    'Iris-setosa':'Iris-setosa',
    #    'Iris-versicolor':'Iris-versicolor',
    #    'Iris-virginica':'Iris-virginica',
    #}

    def __init__( self ):
        ''' A list of classes '''
        self.__classes = None

        #print ",".join(sorted(self.TRAINING_ATTACK_TYPES.keys()))
        #stop

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

    def read ( self, header_file, data_file ):
        ''' Loads the data '''

        self.__read_header( header_file )
        self.__read_data( data_file )

    def get_nr_features( self ):
        return ( sum( map( len, self.__features[2] ) ) + self.__nr_continuous )

    def get_nr_points( self ):
        ''' Return the total number of points in this dataset '''
        return self.__nr_points

    def get_point( self, index ):
        ''' Returns the data point at index '''
        return self.__data[index].toarray().reshape( -1 )

    def get_sparse_point( self, index ):
        ''' Returns the data point at index '''
        return self.__data[index, :]

    def get_label( self, index ):
        ''' Returns the label for the point at index '''
        return self.__labels[index]

    def get_classes ( self ):
        ''' Returns a list of classes '''
        return self.__classes

    def __read_header( self, names ):
        self.__classes = list()
        self.__features = list()
        self.__nr_continuous = 0
        self.__nr_symbolic = 0

        self.__classes = names.readline().strip( ".\n" ).split( ',' )
        self.__classes = ['normal', 'r2l', 'u2r', 'probe', 'dos']
        for line in names:
            key_value = line.strip( ".\n" ).split( ':' )
            if key_value[1] == 'continuous':
                self.__features.append( ( key_value[0], key_value[1] ) )
                self.__nr_continuous += 1
            else:
                self.__features.append( ( key_value[0], key_value[1].strip( 
                                            ".\n" ).split( ',' ) ) )
                self.__nr_symbolic += 1

        logging.info( '  Read {0} features from file'.
                      format( self.__nr_continuous + self.__nr_symbolic ) )
        logging.info( '    of which {0} are symbolic and {1} are continuous'.
                      format( self.__nr_symbolic, self.__nr_continuous ) )

    def __read_data( self, data_file ):
        ''' Read the actual data '''

        logging.info( '  Read data from file.' )
        self.__data = np.loadtxt( data_file, delimiter=',' )
        #self.__has_labels = ( self.__nr_continuous +
        #                      self.__nr_symbolic ) < self.__data.shape[1]
        self.__has_labels = True
        self.__nr_points = self.__data.shape[0]

        logging.info( '  Read {0} connections from file'.
                      format( self.__nr_points ) )

        if self.__has_labels:
            logging.info( '    labels are present' )
            self.__labels = list()
            keys = self.TRAINING_ATTACK_TYPES.keys()
            keys.sort()

            for i in self.__data[:, -1]:
                self.__labels.append( self.TRAINING_ATTACK_TYPES[keys[int( i )]] )

            # Strip labels
            self.__data = self.__data[:, 0:-1]
        else:
            logging.info( '    labels are not present' )

        logging.info( '  Converting to sparse format' )
        self.__data = sparse.csr_matrix( self.__data, dtype=np.float )
        self.__data.eliminate_zeros()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
