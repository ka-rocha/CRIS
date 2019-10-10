#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Katie Breivik (2017) OLD FROM COSMIC
#
# This file is part of cris
#
# cris is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cris.  If not, see <http://www.gnu.org/licenses/>

"""CRIS
"""

from ._version import get_versions
__version__ = get_versions()['version']
__author__ = 'Kyle Rocha <kylerocha2024@u.northwestern.edu>'
__credits__ = ['Scott Coughlin <scottcoughlin2014@u.northwestern.edu>',
                'Pablo Marchant <pamarca@gmail.com>',
                'Christopher Berry <christopher.berry@northwestern.edu>',
                'Vicky Kalogera <vicky@northwestern.edu>']
del get_versions

# generally useful imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
