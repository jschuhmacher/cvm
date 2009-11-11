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
Cerberus is prototype analysis and detection module for an intrusion detection 
system. It uses the Core Vector Machine (CVM) for analysis. 
'''

# System
import sys
import traceback
import logging
import cPickle as pickle

# 3rd-party
import argparse

# Local
import common.data
import common.model
import train.trainer
import classify.predictor

def data_collection( header_file, data_file ):
    ''' Collect data '''
    logging.info ( 'Data collection phase...' )
    dataset = common.data.Dataset()
    dataset.read( header_file, data_file )
    return dataset

def feature_selection( dataset ):
    ''' In the future do some feature selection '''
    logging.info ( 'Feature selection phase...' )
    logging.info ( '  Not yet implemented' )
    return dataset

def prepare_training( dataset, C, EPS, kernel, gamma ):
    ''' Prepare for training '''
    logging.info ( 'Prepare for training phase...' )
    trainer = train.trainer.Trainer( dataset, C , EPS, kernel, gamma )
    return trainer

def do_training( trainer ):
    ''' Construct classifiers from the training set'''
    logging.info ( 'Training phase...' )
    models = trainer.train()
    return models

def load_models( model_file ):
    ''' Load trained models from file '''
    logging.info ( 'Loading models...' )
    models = pickle.load( model_file )
    return models

def save_models( models, model_file ):
    ''' Save trained models to file '''
    logging.info ( 'Saving models...' )
    pickle.dump( models, model_file, pickle.HIGHEST_PROTOCOL )

def prepare_testing( models, dataset ):
    ''' Initialisation for the testing phase '''
    logging.info ( 'Prepare for testing phase...' )
    predictor = classify.predictor.Predictor( models, dataset )
    return predictor

def do_testing( predictor ):
    ''' Do the testing phase '''
    logging.info ( 'Testing phase...' )
    predictor.concurrent_predict()
    #predictor.predict()

def initialise_cli ():
    ''' Set up command line arguments '''
    parser = argparse.ArgumentParser( description=__doc__ )

    parser.add_argument( 'header', type=argparse.FileType( 'r' ),
                         help='Read feature information from this file.' )

    parser.add_argument( 'dataset', type=str,
                         help='Read data from this file.' )

    parser.add_argument( '--log', type=argparse.FileType( 'a' ),
                         default=sys.stdout,
                         help='Send log output to this file \
                               (defaults to stdout)' )

    parser.add_argument( '--C', type=float,
                         default=1e6,
                         help='Complexity constant (defaults to 1000000)' )

    parser.add_argument( '--EPS', type=float,
                         default=1e-6,
                         help='Epsilon, defaults to 1e-6' )

    parser.add_argument( '--gamma', type=float,
                         default=0.1,
                         help='Gamma for RBF kernel' )

    parser.add_argument( '--kernel', type=int,
                         default=1,
                         help='Kernel type: 0 -> dot, 1 -> RBF' )

    group = parser.add_mutually_exclusive_group( required=True )
    group.add_argument( '--model_out', type=argparse.FileType( 'w' ),
                         help='Store trained models in this file.' )

    group.add_argument( '--model_in', type=argparse.FileType( 'r' ),
                         help='Load trained models from this file.' )

    return parser

def initialise_logging ( log_file ):
    ''' Initialise the logging module '''
    logging.basicConfig( level=logging.DEBUG,
                         format='%(asctime)s - %(levelname)-8s - %(message)s',
                         datefmt='%Y-%m-%d %H:%M:%S',
                         stream=log_file, )
    logging.info( 'Initialised Cerberus' )

def main():
    ''' Entrypoint '''
    parser = initialise_cli()
    arguments = parser.parse_args()
    initialise_logging( arguments.log )
    try:
        if arguments.model_in == None and arguments.model_out != None:
            trainset = data_collection( arguments.header, arguments.dataset )
            features = feature_selection( trainset )
            trainer = prepare_training( trainset, arguments.C, arguments.EPS,
                                        arguments.kernel, arguments.gamma )
            models = do_training( trainer )
            save_models( models, arguments.model_out )
            arguments.header.seek( 0 )
        elif arguments.model_in != None and arguments.model_out == None:
            models = load_models( arguments.model_in )
            testset = data_collection( arguments.header, arguments.dataset )
            predictor = prepare_testing( models, testset )
            do_testing( predictor )
    except:
        exceptionType, exceptionValue, exceptionTraceback = sys.exc_info()
        logging.error( traceback.format_exc() )

if __name__ == '__main__':
    sys.exit( main() )
