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
File: model.py

moduledocs
'''

class Model( object ):
    ''' classdocs
    '''

    class_name = None
    weights = None
    b = None
    examples = None

    def __init__( self, class_name ):
        self.class_name = class_name
