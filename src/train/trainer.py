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
File: trainer.py

moduledocs
'''

# System
import logging
import multiprocessing

# Local
import common.data
import common.model
import cvm

class Trainer( object ):
    ''' classdocs
    '''
    dataset = None
    C = 1e6
    EPS = 1e-6
    gamma = 2.0
    sigma = 300.0
    kernel = 0

    def __init__( self, dataset, C, EPS, kernel, gamma ):
        self.dataset = dataset
        self.C = C
        self.EPS = EPS
        self.kernel = kernel
        self.gamma = gamma

    def train_tasklet ( self, label ):
        ''' Train one classifier for one class '''
        logging.info( "  Starting training job for class <{0}>".format( label ) )
        classifier = cvm.CVM( label, self.dataset, self.EPS, self.kernel, self.C, self.gamma )
        return ( label, classifier.train() )

    def train( self ):
        ''' Train a classifier for every class, tries to train the profiles 
            concurrently.
        '''
        #classes = zip( [self] * len( self.dataset.get_classes() ), self.dataset.get_classes() )
        #pool = multiprocessing.Pool( processes=len( classes ) )
        # Workaround python bug: http://www.mail-archive.com/python-list@python.org/msg228648.html
        #result = pool.map( train_tasklet_mp, classes )

        result = []
        for i in self.dataset.get_classes():
            if i == 'normal':
                result.append( self.train_tasklet( i ) )

        models = {}
        for model in result:
            models[model[0]] = model[1]
        return models

def train_tasklet_mp( ar, **kwar ):
    return Trainer.train_tasklet( *ar, **kwar )

