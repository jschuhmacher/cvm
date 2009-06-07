#!/usr/bin/env python
# vim: set fileencoding=utf-8 ts=4 sw=4 expandtab:

################################################################################
# File: cerberus.py
#
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

The main module of the intrusion detection system.
'''

import sys
import common.data
import common.model
import train.trainer
import logging
import argparse

def data_collection( data_file ):
    logging.info ( 'Data collection phase...' )
    dataset = common.data.Dataset()
    dataset.read( data_file )
    return dataset

def feature_selection( dataset ):
    logging.info ( 'Feature selection phase...' )
    logging.info ( '   Not yet implemented' )
    pass

def prepare_training( dataset ):
    logging.info ( 'Prepare for training phase...' )
    trainer = train.trainer.Trainer( dataset )
    return trainer

def do_training( trainer, dataset ):
    logging.info ( 'Training phase...' )
    models = trainer.train()
    return models

def prepare_testing( models, dataset ):
    logging.info ( 'Prepare for testing phase...' )
    logging.info ( '   Not yet implemented' )
    pass

def do_testing( models, dataset ):
    logging.info ( 'Testing phase...' )
    logging.info ( '   Not yet implemented' )
    pass

def initialise_cli ():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--datafile', type=argparse.FileType( 'r' ),
                         default=sys.stdin,
                         help='Read data from this file.\
                               (defaults to stdin)' )

    parser.add_argument( '--logfile', type=argparse.FileType( 'a' ),
                         default=sys.stdout,
                         help='Send log output to this file. \
                               (defaults to stdout)' )

    return parser.parse_args()

def initialise_logging ( log_file ):
    logging.basicConfig( level=logging.DEBUG,
                         format='%(asctime)s - %(levelname)-8s - %(message)s',
                         datefmt='%Y-%m-%d %H:%M',
                         stream=log_file, )
    logging.info( 'Initialised Cerberus' )

def main():
    arguments = initialise_cli()
    initialise_logging( arguments.logfile )
    try:
        dataset = data_collection( arguments.datafile )
        features = feature_selection( dataset )
        trainer = prepare_training( dataset )
        models = do_training( trainer, dataset )
        prepare_testing( models, dataset )
        do_testing( models, dataset )
    except Exception as error:
        logging.error( "{0}: {1}".format( type( error ), error ) )


if __name__ == '__main__':
    sys.exit( main() )
