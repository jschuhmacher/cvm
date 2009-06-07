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

This is the "data" module, it contains data structures for labelled and unlabel-
led sparse data sets for use within a linear classifier. It provides several 
convenience methods, e.g. for IO, splitting, shuffling, normalisation etc. 
'''

from arff import arffread, arffwrite
import logging
import sys
import pprint
import random
import scipy as sp
import scipy.sparse as sparse

class Dataset( object ):
    ''' Data structure representing a dataset.
    '''

    name = None
    sparse = False
    attributes = None

    ''' After reading a file, data has the scipy.sparse.lil_matrix format
    '''
    data = None
    data_labels = None

    classes = None
    nr_points = None

    def read( self, data_file ):
        ''' Reads a data file in the Weka ARFF format. 
            
            *Assumption*:
              The last column is the class label.
            
            First create a dataset object:: 
            >>> d = Dataset()
            
            Then read a data file into that object::
            >>> d.read( 'spambase_fake.arff' )
            ...
            >>> pprint.PrettyPrinter().pprint(d.data)
            <10x58 sparse matrix of type '<type 'numpy.float64'>' 
                with 185 stored elements in LInked List format>
            
            Once read into memory the data will be shuffled and converted to a
            sparse matrix representation for efficiency.
        '''
        logging.info( "  Reading <{0}>".format( data_file.name ) )
        ( self.name, self.sparse, self.attributes, self.data ) = arffread( data_file )
        logging.info( "  Read {0!s} records from <{1}>".format( self.size(), data_file.name ) )

        # Shuffle the data set
        self.shuffle()

        logging.info( "  Converting {0!s} records to sparse format.".format( self.size() ) )
        self.data = sparse.lil_matrix( self.data )

        # Store the labels separately
        self.data_labels = self.data[:, len( self.attributes ) - 1]

        # Strip the labels from the data set
        self.data = self.data.tocsr()[:, 0:len( self.attributes ) - 1]

    def shuffle ( self ):
        ''' Shuffles the data in this dataset '''
        logging.info( "  Shuffling {0!s} records".format( self.size() ) )
        random.shuffle( self.data )

    def split( self, percentage=66.6 ):
        ''' Returns 2 subsets of this dataset according to the parameter
        '''
        pass

    def normalise ( self ):
        ''' Normalises the values in this dataset to the range 0..1 '''
        pass

    def get_point( self, index ):
        ''' Returns one data point '''
        return self.data[index]

    def get_random_point ( self ):
        ''' Returns one random data point. '''
        pass

    def size ( self ):
        ''' Returns the number of data points.  '''
        if self.nr_points is None:
            self.nr_points = len( self.data )
        return self.nr_points

if __name__ == '__main__':
    import doctest
    doctest.testmod()
