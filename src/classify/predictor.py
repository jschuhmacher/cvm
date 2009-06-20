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
File: predictor.py

moduledocs
'''

# System
import pprint
import logging
import multiprocessing

# Local
import common.data
import common.model
import numpy as np

class Predictor( object ):
    ''' 
    classdocs
    '''

    def __init__( self, models, testset ):

        ''' Previous learned models '''
        self.__models = models

        ''' A labelled testset '''
        self.__testset = testset

    def score_point ( self, model, point ):
        ''' Calculate the score of one record '''
        return model.predict( point )

    def concurrent_score_point ( self, point ):
        ''' Concurrently score points '''

        if ( point % 1000 ) == 0 and point > 0:
            logging.debug( '  Scored {0} records'.format( 
                           len( self.__models.keys() ) * 1000 ) )
        true_label = self.__testset.get_label( point )
        x = self.__testset.get_sparse_point( point )

        scores = []
        for label in self.__models.keys():
            scores.append( ( label,
                             self.__models[label].predict( x ),
                             true_label ) )

        return scores

    def concurrent_predict( self ):
        ''' Evaluate the models with the testset '''
        nr_classes = len( self.__models.keys() )
        classes = self.__models.keys()
        nr_points = len( self.__testset.get_indices() )

        pool = multiprocessing.Pool( processes=multiprocessing.cpu_count() )
        logging.debug( '  Scoring {0} records, using {1} CPUs'.format( 
                        nr_points, multiprocessing.cpu_count() )
                     )
        scores = pool.map( self.concurrent_score_point,
                           self.__testset.get_indices(),
                           100 )

        class_statistics = {}
        for label in self.__models.keys():
            class_statistics[label] = {'total': 0,
                                       'tp': 0,
                                       'fp': 0,
                                       'tn':0,
                                       'fn': 0 }

        for score in scores:
            for label, score, true_label in score:
                class_statistics[label]['total'] += 1

                if score >= 0:
                    if label == true_label:
                        class_statistics[label]['tp'] += 1.0
                    if label != true_label:
                        class_statistics[label]['fp'] += 1.0
                if score < 0:
                    if label == true_label:
                        class_statistics[label]['fn'] += 1.0
                    if label != true_label:
                        class_statistics[label]['tn'] += 1.0

        self.print_class_statistics( class_statistics )

        # Python dies without this print statement
        print ''

    def predict ( self ):
        ''' Evaluate the models with the testset '''
        nr_classes = len( self.__models.keys() )
        classes = self.__models.keys()
        nr_total = len( self.__testset.get_indices() )
        nr_scores = 0

        # total, fp, fn, tp
        class_statistics = {}

        for label in self.__models.keys():
            class_statistics[label] = {'total': 0,
                                       'tp': 0,
                                       'fp': 0,
                                       'tn':0,
                                       'fn': 0 }

        for point in self.__testset.get_indices():
            true_label = self.__testset.get_label( point )
            x = self.__testset.get_sparse_point( point )

            for index, label in enumerate( classes ):
                score = self.score_point( self.__models[label], x )
                nr_scores += 1

                class_statistics[label]['total'] += 1

                if score >= 0:
                    if label == true_label:
                        class_statistics[label]['tp'] += 1.0
                    if label != true_label:
                        class_statistics[label]['fp'] += 1.0
                if score < 0:
                    if label == true_label:
                        class_statistics[label]['fn'] += 1.0
                    if label != true_label:
                        class_statistics[label]['tn'] += 1.0

            if ( nr_scores % 1000 ) == 0 and nr_scores > 0:
                logging.debug( '  Calculated 1000 scores' )

        self.print_class_statistics( class_statistics )
        #self.print_total_statistics( class_statistics )

    def print_total_statistics( self, class_statistics ):
        ''' Print total statistics to the screen. '''
        pass

    def print_class_statistics( self, class_statistics ):
        ''' Print some statistics to the screen '''
        print ''
        print 'Class statistics:'
        print ''
        print '{0:<14}|{1:>6}{2:>6}{3:>6}{4:>6} |{5:>8}{6:>8}{7:>8}'.\
              format( 'Pred. class', 'TP', 'TN', 'FP', 'FN', 'ACC', 'TPR',
                      'FPR' )
        print '-----------------------------------------------------------------'
        for label in class_statistics.keys():
            precision = 0.0
            if ( class_statistics[label]['tp'] + class_statistics[label]['fp'] ) > 0:
                precision = ( class_statistics[label]['tp'] /
                              ( class_statistics[label]['tp'] +
                                class_statistics[label]['fp'] ) )

            recall = 0.0
            if ( class_statistics[label]['tp'] + class_statistics[label]['fn'] ) > 0:
                recall = ( class_statistics[label]['tp'] /
                           ( class_statistics[label]['tp'] +
                             class_statistics[label]['fn'] ) )

            fall_out = 0.0
            if ( class_statistics[label]['fp'] + class_statistics[label]['tn'] ) > 0:
                fall_out = ( class_statistics[label]['fp'] /
                             ( class_statistics[label]['fp'] +
                               class_statistics[label]['tn'] ) )

            accuracy = 0.0
            if class_statistics[label]['total'] > 0:
                accuracy = ( class_statistics[label]['tp'] +
                             class_statistics[label]['tn'] ) / \
                             class_statistics[label]['total']

            f1 = 0.0
            if ( precision + recall ) > 0:
                f1 = ( 2.0 * precision * recall ) / ( precision + recall )

            print '{0:<14}|{1:>6}{2:>6}{3:>6}{4:>6} |{5:>8.4}{6:>8.4}{7:>8.4}'.\
                   format( label,
                           int( class_statistics[label]['tp'] ),
                           int( class_statistics[label]['tn'] ),
                           int( class_statistics[label]['fp'] ),
                           int( class_statistics[label]['fn'] ),
                           accuracy,
                           recall,
                           fall_out )

    def print_confusion_matrix( self, classes, confusion ):
        ''' Print a confusion matrix to the screen '''

        print ''
        print 'Confusion matrix:'
        print ''
        print '{0:>8} | '.format( ' ' ),
        for actual, alabel in enumerate( classes ):
            print '{0:>8}'.format( alabel ),
        print ''
        print '--------------------------------------------------------'

        for pred, plabel in enumerate( classes ):
            print '{0:<8} | '.format( plabel ),
            for actual, alabel in enumerate( classes ):
                print '{0:>8}'.format( int( confusion[pred, actual] ) ),
            print ''

        print ''
        print '* Columns are truth, rows are predicted.'
        print ''

if __name__ == '__main__':
    import doctest
    doctest.testmod()
