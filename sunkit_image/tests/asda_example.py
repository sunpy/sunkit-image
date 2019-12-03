#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Nov 25 2019

Name: asda.py

Purpose: An example of using ASDA

@author: Jaijia Liu at University of Sheffield
"""
__author__ = 'Jiajia Liu'
__copyright__ = 'Copyright 2019, The Solar Physics and Space Plasma ' + \
                'Research Center (SP2RC)'
__license__ = 'GPLv3'
__date__ = '2019/11/25'
__maintainor__ = 'Jiajia Liu'
__email__ = 'jj.liu@sheffield.ac.uk'


import numpy as np
from datetime import datetime
import requests
import astropy

def asda_example():
    """
    run the example on real data and compare with the correct answer

    Notes:
        Input velocity field and image (if there is any) are all stored in
        default Python order (i.e. [y, x] of the data).

        Output gamma values are in the same order, thus the same shape as
        velocity field.

        other outputs are in the order of [x, y], i.e., vc = [vx, vy],
        edge = [[x1, y1], [x2, y2],...], points = [[x1, y1], [x2, y2],...]
        in units of pixel
    """
    # file which stores the velocity field data
    vel_file = astropy.utils.data.download_file(
               "https://github.com/PyDL/asda-class/blob/master/data/vxvy.npz")
    # file that stores the correct detection result
    cor_file = astropy.utils.data.download_file(
               "https://github.com/PyDL/asda-class/blob/master/data/correct.npz")
    # load velocity field and data
    vxvy = np.load(vel_file)
    vx = vxvy['vx']
    vy = vxvy['vy']
    data = vxvy['data']
    
    # Perform swirl detection
    factor = 1
    # Initialise class
    lo = asda.Asda_Calc(vx, vy, factor=factor)
    # Gamma1 and Gamma2
    beg_time = datetime.today()
    gamma = lo.gamma_values()
    # Caculate time consumption
    end_time = datetime.today()
    print('Time used for calculating Gamma1 & Gamma2', end_time-beg_time)
    # Determine Swirls
    center_edge = lo.center_edge()
    # Properties of Swirls
    ve, vr, vc, ia = lo.vortex_property(image=data)
    # load correct detect results
    correct = dict(np.load(cor_file, allow_pickle=True))
    
    # visualise gamma2
    lo.visual_gamma(gamma2=True)
    
    # compare between detection result and correct detection result
    # number of swirls
    n = len(ve)
    nc = len(correct['ve'])
    if n != nc:
        raise Exception("The number of swirls is wrong!")
    
    # find correspondances
    pos = []
    i = 0
    for cen in center_edge['center']:
        cen = [int(cen[0]), int(cen[1])]
        idx = np.where(correct['center'] == cen)
        if np.size(idx[0]) < 2:
            raise Exception("At least one swirl is not in the correct" +
                            " position")
        pos.append(np.bincount(idx[0]).argmax())
    
    # perform comparison
    peak_diff = []
    radius_diff = []
    vr_diff = []
    ve_diff = []
    vc_diff = []
    ia_diff = []
    for i in np.arange(n):
        idx = pos[i]
        peak_diff.append((center_edge['peak'][i] - correct['peak'][idx]) /
                         correct['peak'][idx])
        radius_diff.append((center_edge['radius'][i] -
                            correct['radius'][idx]) / correct['radius'][idx])
        vr_diff.append((vr[i] - correct['vr'][idx]) / correct['vr'][idx])
        ve_diff.append((ve[i] - correct['ve'][idx]) / correct['ve'][idx])
        vc_diff.append((vc[i] - correct['vc'][idx]) / correct['vc'][idx])
        ia_diff.append((ia[i] - correct['ia'][idx]) / correct['ia'][idx])

    print('Difference in Peak Gamma1 Value:', np.max(peak_diff),
          np.mean(peak_diff), np.min(peak_diff))
    print('Difference in radius:', np.max(radius_diff),
          np.mean(radius_diff), np.min(radius_diff))
    print('Difference in rotating speed:', np.max(vr_diff),
          np.mean(vr_diff), np.min(vr_diff))
    print('Difference in expanding speed:', np.max(ve_diff),
          np.mean(ve_diff), np.min(ve_diff))
    print('Difference in average intensity:', np.max(ia_diff),
          np.mean(ia_diff), np.min(ia_diff))

