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
import stackless
import logging

# Local
import common.data
import common.model
import cvm

class Trainer( object ):
    ''' classdocs
    '''
    dataset = None
    models = None

    def __init__( self, dataset ):
        self.dataset = dataset
        self.models = dict()

    def train_tasklet ( self, label ):
        ''' Train one classifier for one class '''
        classifier = cvm.CVM( label, self.dataset )
        self.models[label] = classifier.train()

    def train( self ):
        ''' Train a classifier for every class, tries to train the profiles 
            concurrently.
        '''
        for label in self.dataset.get_classes():
            logging.info( "  Initialising training job for class <{0}>".
                          format( label ) )
            stackless.tasklet( self.train_tasklet )( label )


        logging.info( "  Scheduling training jobs" )
        stackless.schedule()
        logging.info( '  Starting training jobs' )
        stackless.run()

        return self.models

