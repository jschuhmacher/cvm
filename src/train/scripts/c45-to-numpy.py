#!/usr/bin/env python2.6

from numpy import *
import argparse

def read_names( filename ):
    classes = list()
    features = list()

    with open( filename + '.names', 'r' ) as names:
        classes = names.readline().strip( ".\n" ).split( ',' )
        for line in names:
            key_value = line.strip( ".\n" ).split( ':' )
            if key_value[1] == 'continuous':
                features.append( ( key_value[0], key_value[1] ) )
            else:
                features.append( ( key_value[0], key_value[1].strip( ".\n" ).split( ',' ) ) )

    return ( classes, features )

def read_data( filename, classes, features ):
    nr_features = len( features )
    counter = 0

    nr_total_features = 1
    for feature in features:
        if feature[1] == 'continuous':
            nr_total_features += 1
        else:
            nr_total_features += len( feature[1] )

#    data = None
    data = []

    with open( filename + '.data', 'r' ) as datafile:
        for line in datafile:
            record = line.strip( ".\n" ).split( ',' )
            numeric_record = zeros( nr_total_features, dtype=float32 )

            index1 = 0
            for index2, feature in enumerate( record ):

                if index2 < nr_features and features[index2][1] == 'continuous':
                    numeric_record[index1] = feature
                    index1 += 1
                elif index2 == nr_features:
                    if feature not in classes:
                        classes.append(feature)
                        print feature
                    numeric_record[index1] = classes.index( feature )
                    index1 += 1
                else:
                    for indicator in features[index2][1]:
                        if feature == indicator:
                            numeric_record[index1] = 1.
                        else:
                            numeric_record[index1] = 0.
                        index1 += 1
                    func = None

#            if data == None:
#                data = numeric_record
#            else:
#                data = vstack( (data, numeric_record) )
            data.append(numeric_record)
            counter += 1
            record = None
            if ( counter % 10000 ) == 0:
                print "{0:>6.4} % of lines read...".format( ( counter / 4898431. ) * 100. )
    
    return data

def normalise( data ):
    maxima = amax(data, axis=0)
    print maxima
    maxima = maxima + select( [maxima == 0, True], [1., 0.] )
    maxima[-1] = 1.
    print maxima
    data = data /  maxima
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser( description=__doc__ )
    parser.add_argument( 'dataset_in', type=str,
                     help='Read data from this file.' )

    parser.add_argument( 'dataset_out', type=str,
                     help='Write data to this file.' )

    arguments = parser.parse_args()
    filename = arguments.dataset_in
    filename_out = arguments.dataset_out

    print "Read names.."
    classes, features = read_names( filename )
    print "Read data.. & convert to matrix"
    data = read_data( filename, classes, features )    
    data = array(data)
    print "Save array.."
    savetxt( filename_out + '.txt.gz', data, '%.10g', ',' )
    print "Normalise data..."
    data = normalise( data )
    print "Save normalised array.."
    savetxt( filename_out + '-normalised.txt.gz', data, '%.10g', ',' )
