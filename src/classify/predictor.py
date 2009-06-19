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

# Local
import common.data
import common.model

# 3rd-party
import psyco
psyco.full()

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

    def predict ( self ):
        ''' Evaluate the models with the testset '''

        nr_classes = len( self.__models.keys() )
        nr_total = len( self.__testset.get_indices() )

        # total, fp, fn, tp
        class_statistics = {}
        for label in self.__models.keys():
            class_statistics[label] = {'total': 0, 'tp': 0, 'fp': 0, 'tn':0,
                                       'fn': 0 }

        nr_scores = 0
        for point in self.__testset.get_indices():
            true_label = self.__testset.get_label( point )
            class_statistics[true_label]['total'] += 1

            x = self.__testset.get_sparse_point( point )

            for label in self.__models.keys():
                score = self.score_point( self.__models[label], x )
                nr_scores += 1

                if score >= 0 and label == true_label:
                    class_statistics[label]['tp'] += 1
                if score >= 0 and label != true_label:
                    class_statistics[label]['fp'] += 1
                if score < 0 and label == true_label:
                    class_statistics[label]['fn'] += 1
                if score < 0 and label != true_label:
                    class_statistics[label]['tn'] += 1


            if ( nr_scores % 1000 ) == 0 and nr_scores > 0:
                logging.debug( '  Calculated 1000 scores' )


        print '{0:<14}|{1:>6}{2:>6}{3:>6}{4:>6} |{5:>8}{6:>8}{7:>8}'.\
              format( 'Pred. class', 'TP', 'TN', 'FP', 'FN', 'Prec.', 'Rec.',
                      'F1' )
        print '-----------------------------------------------------------------'
        for label in class_statistics.keys():
            precision = 0.0
            if ( class_statistics[label]['tp'] + class_statistics[label]['fp'] ) > 0:
                precision = class_statistics[label]['tp'] / ( float( class_statistics[label]['tp'] ) + float( class_statistics[label]['fp'] ) )

            recall = 0.0
            if ( class_statistics[label]['tp'] + class_statistics[label]['fn'] ) > 0:
                recall = class_statistics[label]['tp'] / ( float( class_statistics[label]['tp'] ) + float( class_statistics[label]['fn'] ) )

            f1 = 0.0
            if ( precision + recall ) > 0:
                f1 = ( 2.0 * precision * recall ) / ( precision + recall )

            print '{0:<14}|{1:>6}{2:>6}{3:>6}{4:>6} |{5:>8.4}{6:>8.4}{7:>8.4}'.\
                   format( label, class_statistics[label]['tp'],
                           class_statistics[label]['tn'],
                           class_statistics[label]['fp'],
                           class_statistics[label]['fn'],
                           precision, recall, f1 )

if __name__ == '__main__':
    import doctest
    doctest.testmod()
