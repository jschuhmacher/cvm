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

import common.data
import common.model
import numpy

class Predictor( object ):
    ''' classdocs
    '''

    def score ( self, model, xs ):
        ''' Calculate the score of multiple records: score = w^T * x + b '''
        scores = []
        for class_profile in self.model.profiles:
            scores.append( numpy.linalg.dot( class_profile.weights, xs ) + class_profile.b )
        return scores

    def evaluate ( self, model, xs ):
        ''' Evaluate the model with the testset xs '''
        pass

if __name__ == '__main__':
    import doctest
    doctest.testmod()
