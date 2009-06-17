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

# System
import sys
import traceback
import logging

# 3rd-party
import argparse

# Local
import common.data
import common.model
import train.trainer

def data_collection( header_file, data_file ):
    logging.info ( 'Data collection phase...' )
    dataset = common.data.Dataset()
    dataset.read( header_file, data_file )
    return dataset

def feature_selection( dataset ):
    logging.info ( 'Feature selection phase...' )
    logging.info ( '  Not yet implemented' )
    return dataset

def prepare_training( dataset ):
    logging.info ( 'Prepare for training phase...' )
    dataset.preprocess()
    dataset.normalise()
    dataset.shuffle()
    trainer = train.trainer.Trainer( dataset )
    return trainer

def do_training( trainer ):
    logging.info ( 'Training phase...' )
    models = trainer.train()
    return models

def prepare_testing( models, dataset ):
    logging.info ( 'Prepare for testing phase...' )
    logging.info ( '  Not yet implemented' )
    pass

def do_testing( models, dataset ):
    logging.info ( 'Testing phase...' )
    logging.info ( '  Not yet implemented' )
    pass

def initialise_cli ():
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( '--data', type=argparse.FileType( 'r' ),
                         default=sys.stdin,
                         help='Read data from this file.\
                               (defaults to stdin)' )

    parser.add_argument( '--header', type=argparse.FileType( 'r' ),
                         required=True,
                         help='Read feature information from this file.' )

    parser.add_argument( '--log', type=argparse.FileType( 'a' ),
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
    initialise_logging( arguments.log )
    try:
        dataset = data_collection( arguments.header, arguments.data )
        features = feature_selection( dataset )
        trainer = prepare_training( dataset )
        models = do_training( trainer )
        prepare_testing( models, dataset )
        do_testing( models, dataset )
    except:
        exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
        logging.error( traceback.format_exc() )


if __name__ == '__main__':
    sys.exit( main() )
