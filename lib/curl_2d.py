#!/usr/bin/python -tt
#=======================================================================
#                        General Documentation

"""Single-function module.

   See function docstring for description.
"""

#-----------------------------------------------------------------------
#                       Additional Documentation
#
# RCS Revision Code:
#   $Id: curl_2d.py,v 1.11 2004/05/06 23:33:37 jlin Exp $
#
# Modification History:
# - 01 Mar 2004:  Original by Johnny Lin, Computation Institute,
#   University of Chicago.  Passed passably reasonable tests.
# - 04 May 2004:  Add non-scalar R_sphere capability.  Passed
#   passably reasonable tests.
#
# Notes:
# - Written for Python 2.2.
# - Module docstrings can be tested using the doctest module.  To
#   test, execute "python curl_2d.py".  Additional tests can be
#   found in the test sub-directory of the distribution (see the
#   package home page below for information).
# - Module has a single overall function, within which there are 
#   several private level-1 nested functions that define different 
#   methods of computing curl.
# - See import statements throughout module for packages and modules 
#   required.
#
# Copyright (c) 2004 by Johnny Lin.  For licensing, distribution 
# conditions, contact information, and additional documentation see
# the URL http://www.johnny-lin.com/py_pkgs/gemath/doc/.
#=======================================================================




#----------------------- Overall Module Imports ------------------------

#- The "from __future__ import" line as the first line in the module
#  enables nested-scope access for Python versions prior to 2.2.
#  From 2.2 and on nested-scope access is automatic:

from __future__ import nested_scopes


#- Set module version to package version:

import gemath_version
__version__ = gemath_version.version
__author__  = gemath_version.author
__date__    = gemath_version.date
__credits__ = gemath_version.credits
del gemath_version




#------------- Overall Function:  Documentation and Import -------------

def curl_2d( x, y, Fx, Fy, missing=1e+20, algorithm='default', R_sphere=6.37122e+6):
    """Curl of a vector F on a 2-D "rectangular" grid.

    The 2-D grid F is defined on is rectangular, meaning that while
    the grid spacings do not have to be even, each grid box is
    rectangular and grid boundaries are rectilinear and parallel to
    each other.  If the 2-D grid is on a sphere, we assume that lines
    of constant longitude and latitude form grids that are rectangular
    (though in reality this is not strictly true).

    Since F does not have a z-component, the curl of F is positive
    pointing as a normal out of the plane defined by the basis for 
    x and y.


    Positional Input Arguments:
    * x:  x-coordinate of each F.  1-D Numeric array with number of 
      elements equal to the number of columns in array F.  Typically 
      x is in km, or on a sphere the longitude in deg.  Floating or
      integer type.

    * y:  y-coordinate of each F.  1-D Numeric array with number of 
      elements equal to the number of rows in array F.  Typically 
      y is in km, or on a sphere the latitude in deg.  Floating or
      integer type.

    * Fx:  x-component of vector quantity F.  2-D Numeric array of 
      shape (len(y), len(x)).  Floating or integer type.

    * Fy:  y-component of vector quantity F.  2-D Numeric array of 
      same shape and size as Fx.  Floating or integer type.


    Keyword Input Arguments:
    * missing:  If Fx and/or Fy has missing values, this is the 
      missing value value.  Scalar.  Default is 1e+20.

    * algorithm:  Name of the algorithm to use.  String scalar.
      Default is 'default'.  Possible values include:

      + 'default':  Default method.  Algorithm 'order1_cartesian' 
        is used.

      + 'default_spherical':  If 'spherepack' can be used, chooses
        that method.  Otherwise 'order1_spherical' is used.  Either
        way this option assumes the domain is on a sphere of radius
        equal to keyword R_sphere.

      + 'order1_cartesian':  First-order finite-differencing (back-
        ward and forward differencing used at the endpoints, and 
        centered differencing used everywhere else).  If abscissa 
        intervals are irregular, differencing will be correspon-
        dingly asymmetric.  Grid is assumed Cartesian.  The curl
        returned is in units of F divided by the units of x and y
        (x and y must be in the same units).

      + 'order1_spherical':  First-order finite-differencing (back-
        ward and forward differencing used at the endpoints, and 
        centered differencing used everywhere else).  If abscissa 
        intervals are irregular, differencing will be correspon-
        dingly asymmetric.  Grid is assumed spherical and x and y
        are longitude and latitude (in deg).  The curl returned is 
        in units of F divided by units of R_sphere.  Note because
        of singularities in the algorithm near the poles, the
        curl for all points north of 88N or south of 88S latitude 
        are set to missing.

      + 'spherepack':  The NCAR package SPHEREPACK 3.0 is used for 
        calculating the curl.  The spatial domain is the global 
        sphere; x and y are longitude and latitude (in deg), 
        respectively; x must be evenly spaced and y must be gauss-
        ian or evenly spaced.  The domain must cover the entire 
        sphere and x and y must be longitude and latitude, respec-
        tively, in deg.  There can be no missing values.  The 
        algorithm also assumes that x and y are monotonically in-
        creasing from index 0.  The curl returned is in units of F 
        divided by m.  Thus, if F is velocity in m/s, the curl of 
        F is in units 1/s (and is the vorticity).  Note that in 
        this function's implementation, the final curl output using 
        algorithm 'spherepack' is adjusted for R_sphere not equal 
        to the default mean earth radius, if appropriate, so this 
        algorithm can be used on other planets.

        Note that SPHEREPACK operates on single precision floating
        point (Float32) values.  This function silently converts
        input to Float32 for the computations, as needed (the input 
        parameters are not altered, however).

    * R_sphere:  If the grid is the surface of a sphere, this is 
      the radius of the sphere.  This keyword is only used when 
      the algorithm is 'order1_spherical' or 'spherepack' (either 
      chosen explicitly by the algorithm keyword or automatically 
      when 'default_spherical' is used).  Default value is the 
      "mean" radius of the Earth, in meters.  "Mean" is in quotes
      because this value changes depending on what you consider the
      mean; the default in this function is the same Earth radius
      used in module sphere.  Note that if the level is a distance 
      z above the "mean" radius of the earth, R_sphere should be 
      set to the "mean" radius plus z.  
      
      Value can be a scalar or a Numeric or MA array of the same 
      size and shape as Fx.  If keyword is an array, the value of
      R_sphere for each element corresponds to the same location
      as in the Fx and Fy arrays; there should also be no missing 
      values in R_sphere (function doesn't check for this, however).


    Output:
    * Curl of F.  Units depend on algorithm used to calculate curl
      (see above discussion of keyword algorithm).  Numeric array of 
      same shape as Fx and Fy inputs.  
      
      If there are missing values, those elements in the output that 
      used missing values in the calculation of the curl are set to 
      the value in keyword missing.  If there are missing values in 
      the output due to math errors and keyword missing is set to 
      None, output will fill those missing values with the MA default 
      value of 1e+20.


    References:
    * Glickman, T. S. (Ed.) (2000):  "Curl," Glossary of Meteorology,
      Boston, MA:  American Meteorological Society, ISBN 1-878220-
      34-9.
    * Haltiner, G. J, and R. T. Williams (1980):  Numerical Prediction
      and Dynamic Meteorology, New York:  John Wiley & Sons, pp. 8-9.
    * Holton, J. R. (1992):  An Introduction to Dynamic Meteorology,
      San Diego, CA:  Academic Press, p. 482.
    * Kauffman, B. G., and W. G. Large (2002):  The CCSM Coupler,
      Version 5.0, User's Guide, Source Code Reference, and Scientific
      Description.  Boulder, CO:  NCAR.  Value for earth radius is 
      given at URL:
      http://www.ccsm.ucar.edu/models/ccsm2.0/cpl5/users_guide/node10.html
    * Overview of the CDAT Interface to the NCAR SPHEREPACK 3.0.  
      SPHEREPACK is by J. C. Adams and P. N. Swarztrauber.  More
      information on SPHEREPACK can be found at:  
         http://www.scd.ucar.edu/css/software/spherepack/
      while information on the CDAT implementation used here is at:
         http://esg.llnl.gov/cdat/getting_started/cdat.htm


    Example of a synthetic dataset of winds.  The curl is the vorti-
    city.  Images of the wind field and the vorticity are online at
    http://www.johnny-lin.com/py_pkgs/gemath/doc/test_curl_2d_add.html.
    Velocity at y greater than 60 and less than -60 are set to 0, to
    prevent spectral algorithms from giving errors at the poles.  Tiny 
    values less than 1e-10 are also set to 0:

    (a) Import statements and create data:

    >>> import Numeric as N
    >>> from curl_2d import curl_2d
    >>> nx = 181
    >>> ny = 91
    >>> x = N.arange(nx) * 2.0 - 180.0
    >>> y = N.arange(ny) * 2.0 - 90.0
    >>> y_2d = N.reshape( N.repeat(y, nx), (ny, nx) )
    >>> x_2d = N.reshape( N.repeat(N.reshape(x,(1,nx)), ny), (ny, nx) )
    >>> u = N.sin(x_2d*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> v = N.sin((x_2d-30.)*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> N.putmask( u, N.where(N.absolute(u) < 1e-10, 1, 0), 0. )
    >>> N.putmask( v, N.where(N.absolute(v) < 1e-10, 1, 0), 0. )
    >>> N.putmask( u, N.where(N.absolute(y_2d) > 60., 1, 0), 0. )
    >>> N.putmask( v, N.where(N.absolute(y_2d) > 60., 1, 0), 0. )

    (b) Use order 1 Cartesian algorithm:  If x and y are in meters,
        and u and v are in m/s, the curl is in 1/s:

    >>> curl = curl_2d(x, y, u, v, algorithm='order1_cartesian')
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-0.007251593', '-0.003628007', '0', '0.003628007']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['0', '0', '0', '0']

    (c) Use spectral method from SPHEREPACK (need to first remove the
        repeating dateline point from the domain).  Here we take x and
        y in deg, while u and v are in m/s.  If so the curl is in 1/s.
        These results as much less than for example (b) above since
        the domain is much larger:

    >>> x = x[0:-1]
    >>> u = u[:,0:-1]
    >>> v = v[:,0:-1]
    >>> curl = curl_2d(x, y, u, v, algorithm='spherepack')
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-6.436402e-08', '-3.220152e-08', '1.33852e-13', '3.220165e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['-2.299736e-20', '-4.806512e-21', '-9.283145e-22', '-1.368445e-20']

    (d) Use order 1 spherical algorithm:  x and y are in deg and u and
        v are in m/s.  The curl is in 1/s.  Note that these results are
        nearly identical to those from SPHEREPACK, except near the poles
        the values are set to missing:

    >>> curl = curl_2d(x, y, u, v, algorithm='order1_spherical')
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-6.517317e-08', '-3.260645e-08', '0', '3.260645e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['1e+20', '1e+20', '1e+20', '1e+20']

    (e) Use "default" algorithm, which for this domain chooses the 
        order 1 cartesian algorithm:

    >>> curl = curl_2d(x, y, u, v, algorithm='default')
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-0.007251593', '-0.003628007', '0', '0.003628007']

    (f) Use "default_spherical" algorithm with missing data (equal to
        1e+20), which because of the missing data will choose the order
        1 spherical algorithm.  We also do the calculation on Mars, and
        illustrate passing in the radius as a constant as well as an
        array of the same size and shape as u and v:

    >>> u[30:46,90:114] = 1e+20
    >>> curl = curl_2d( x, y, u, v, algorithm='default_spherical' \
                      , R_sphere=3390e+3)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-1.224875e-07', '-6.128107e-08', '1e+20', '1e+20']

    >>> R_mars = N.zeros(u.shape, typecode=N.Float) + 3390e+3
    >>> curl = curl_2d( x, y, u, v, algorithm='default_spherical' \
                      , R_sphere=R_mars )
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-1.224875e-07', '-6.128107e-08', '1e+20', '1e+20']
    """
    import MA
    import Numeric as N
    from has_close import has_close
    from can_use_sphere import can_use_sphere



#------- Nested Function:  Curl By Order 1 Cartesian Finite Diff. ------

    def _order1_cartesian_curl():
        """
        Calculate curl using Cartesian first-order differencing.

        An MA array is returned.

        Algorithm:  First-order differencing (interior points use 
        centered differencing, and end points use forward or backward 
        differencing, as applicable) in Cartesian coordinates (see 
        Glickman [2000], p. 194).
        """
        from deriv import deriv

        dFy_dx_N = N.zeros( (len(y), len(x)), typecode=N.Float )
        dFx_dy_N = N.zeros( (len(y), len(x)), typecode=N.Float )

        for iy in range(len(y)):
            dFy_dx_N[iy,:] = deriv( x, N.ravel(Fy[iy,:]) \
                                  , missing=missing, algorithm='order1')

        for ix in range(len(x)):
            dFx_dy_N[:,ix] = deriv( y, N.ravel(Fx[:,ix]) \
                                  , missing=missing, algorithm='order1')

        dFy_dx = MA.masked_values(dFy_dx_N, missing, copy=0)
        dFx_dy = MA.masked_values(dFx_dy_N, missing, copy=0)

        return dFy_dx - dFx_dy




#------- Nested Function:  Curl By Order 1 Spherical Finite Diff. ------

    def _order1_spherical_curl():
        """
        Calculate curl using spherical first-order differencing.

        An MA array is returned.

        Algorithm:  First-order differencing (interior points use 
        centered differencing, and end points use forward or backward 
        differencing, as applicable) in spherical coordinates (see 
        Holton 1992).

        Note because of singularities in the algorithm near the poles, 
        the return value for all points north of 88N or south of 88S 
        latitude are set to mask true.

        Key to some variables:
        * xrad is x in radians (x is assumed in deg)
        * yrad is y in radians (y is assumed in deg)
        * ddFy is d(Fy) / d(xrad) as a masked array.  The version with
          a "_N" suffix means the Numeric version.
        * ddFx is d(Fx * cos(yrad)) / d(yrad) as a masked array.  The 
          version with a "_N" suffix means the Numeric version.
        """
        from deriv import deriv

        xrad = x * N.pi/180.0
        yrad = y * N.pi/180.0


        #- Derivative preliminaries:

        ddFy_N = N.zeros( (len(yrad), len(xrad)), typecode=N.Float )
        ddFx_N = N.zeros( (len(yrad), len(xrad)), typecode=N.Float )

        for iy in range(len(yrad)):
            ddFy_N[iy,:] = deriv( xrad, N.ravel(Fy[iy,:]) \
                                , missing=missing, algorithm='order1')

        for ix in range(len(xrad)):
            tmp_MA = MA.masked_values(N.ravel(Fx[:,ix]), missing, copy=0)
            tmp_N  = MA.filled(tmp_MA * N.cos(yrad), missing)
            ddFx_N[:,ix] = deriv( yrad, tmp_N \
                                , missing=missing, algorithm='order1')

        ddFy = MA.masked_values(ddFy_N, missing, copy=0)
        ddFx = MA.masked_values(ddFx_N, missing, copy=0)


        #- Calculate the curl:

        tmpr = ( ddFy - ddFx ) / \
               ( R_sphere * N.reshape( N.repeat(N.cos(yrad),len(xrad)) \
                                     , (len(yrad),len(xrad)) ) )


        #- Make points "near" poles be missing and return:

        np_mask_lat =  88.0
        sp_mask_lat = -88.0
        yarr = N.reshape( N.repeat(y,len(x)), (len(y),len(x)) )

        np_mask = MA.make_mask( N.where(yarr > np_mask_lat, 1, 0) )
        sp_mask = MA.make_mask( N.where(yarr < sp_mask_lat, 1, 0) )

        tmpr_mask = tmpr.mask()
        tmpr_mask = MA.mask_or(tmpr_mask, np_mask)
        tmpr_mask = MA.mask_or(tmpr_mask, sp_mask)

        return MA.masked_array(tmpr, mask=tmpr_mask)




#--------- Nested Function:  Curl By SPHEREPACK Spectral Method --------

    def _spherepack_curl():
        """
        Calculate curl using SPHEREPACK spectral methods.
        
        If the value of R_sphere is not the same as the earth radius 
        used in package sphere, adjust curl to reflect R_sphere.  A 
        Numeric array is returned.

        SPHEREPACK outputs a number of diagnostic messages to
        stdout (and perhaps to stderr).  The one that's irritating
        is the one saying data will be converted to Float32.  Thus,
        we silently ensure that calculations are used only on Float32 
        data, to surpress the warnings.  I did this because redirect-
        ing stdout/stderr turned out to be too difficult; just
        altering the sys attributes didn't work.
        """
        import sphere, MA

        sph_obj = sphere.Sphere( x.astype(MA.Float32) \
                               , y.astype(MA.Float32) )
        if N.allclose(N.array(R_sphere), sphere.radius):
            curl = sph_obj.vrt( Fx.astype(MA.Float32) \
                              , Fy.astype(MA.Float32) )
        else:
            curl = sph_obj.vrt( Fx.astype(MA.Float32) \
                              , Fy.astype(MA.Float32) ) \
                 * sphere.radius / R_sphere

        return curl




#-------------- Overall Function:  Settings, Body, Return --------------

    #- Choose algorithm to compute curl:
    
    if algorithm == 'default':
        _calculate_curl = _order1_cartesian_curl

    elif algorithm == 'default_spherical':
        if (can_use_sphere(x,y)[0] == 1) and \
           (not has_close(Fx, missing)) and \
           (not has_close(Fy, missing)):
            _calculate_curl = _spherepack_curl
        else:
            _calculate_curl = _order1_spherical_curl

    elif algorithm == 'order1_cartesian':
        _calculate_curl = _order1_cartesian_curl

    elif algorithm == 'order1_spherical':
        _calculate_curl = _order1_spherical_curl

    elif algorithm == 'spherepack':
        if has_close(Fx, missing) or has_close(Fy, missing):
            raise ValueError, "curl_2d:  has missing values"
        else:
            _calculate_curl = _spherepack_curl

    else:
        raise ValueError, "curl_2d:  bad algorithm"


    #- Calculate curl and return from function:

    return MA.filled( _calculate_curl(), missing )




#-------------------------- Main:  Test Module -------------------------

#- Define additional examples for doctest to use:

__test__ = { 'Additional Examples':
    """
    >>> import Numeric as N
    >>> from curl_2d import curl_2d
    >>> nx = 181
    >>> ny = 91
    >>> x = N.arange(nx) * 2.0 - 180.0
    >>> y = N.arange(ny) * 2.0 - 90.0
    >>> y_2d = N.reshape( N.repeat(y, nx), (ny, nx) )
    >>> x_2d = N.reshape( N.repeat(N.reshape(x,(1,nx)), ny), (ny, nx) )
    >>> u = N.sin(x_2d*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> v = N.sin((x_2d-30.)*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> x = x[0:-1]
    >>> u = u[:,0:-1]
    >>> v = v[:,0:-1]

    (i) Use SPHEREPACK on Mars:

    >>> curl = curl_2d(x, y, u, v, algorithm='spherepack', R_sphere=3390e+3)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-1.213008e-07', '-6.068711e-08', '2.165924e-13', '6.068734e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['-1.150181e-20', '1.700353e-20', '1.227383e-20', '-2.283483e-20']
    >>> ['%.7g' % curl[-1,i] for i in range(10,14)]
    ['1.596643e-20', '5.265175e-21', '4.54957e-21', '-1.17195e-20']
    >>> ['%.7g' % curl[70,i] for i in range(130,134)]
    ['1.474248e-06', '1.470224e-06', '1.464409e-06', '1.456811e-06']

    (ii) Use order 1 spherical on Mars:

    >>> curl = curl_2d(x,y,u,v,algorithm='order1_spherical',R_sphere=3390e+3)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-1.224875e-07', '-6.128107e-08', '-9.383406e-23', '6.128107e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['1e+20', '1e+20', '1e+20', '1e+20']
    >>> ['%.7g' % curl[-1,i] for i in range(10,14)]
    ['1e+20', '1e+20', '1e+20', '1e+20']
    >>> ['%.7g' % curl[70,i] for i in range(130,134)]
    ['1.413254e-06', '1.408895e-06', '1.402819e-06', '1.395035e-06']

    (iii) Use SPHEREPACK with a bad spatial domain (no poles):

    >>> y2 = y[1:-1]
    >>> curl = curl_2d(x, y2, u, v, algorithm='spherepack')
    Traceback (most recent call last):
        ...
    ValueError

    (iv) Use SPHEREPACK with a bad spatial domain (repeating longitude):

    >>> import MA
    >>> x2 = MA.concatenate([x, [x[0]+360.]])
    >>> curl = curl_2d(x, y2, u, v, algorithm='spherepack')
    Traceback (most recent call last):
        ...
    ValueError

    (v) Use SPHEREPACK with missing values:

    >>> u2 = u
    >>> u2[3:5,5:8] = 1e+20
    >>> curl = curl_2d(x, y2, u, v, algorithm='spherepack')
    Traceback (most recent call last):
        ...
    ValueError: curl_2d:  has missing values
    """,


    'Additional Examples for Non-Scalar R_sphere:  Data Set 1':
    """
    >>> import Numeric as N
    >>> from curl_2d import curl_2d
    >>> nx = 181
    >>> ny = 91
    >>> x = N.arange(nx) * 2.0 - 180.0
    >>> y = N.arange(ny) * 2.0 - 90.0
    >>> y_2d = N.reshape( N.repeat(y, nx), (ny, nx) )
    >>> x_2d = N.reshape( N.repeat(N.reshape(x,(1,nx)), ny), (ny, nx) )
    >>> u = N.sin(x_2d*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> v = N.sin((x_2d-30.)*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> x = x[0:-1]
    >>> u = u[:,0:-1]
    >>> v = v[:,0:-1]

    (i) Use SPHEREPACK on Mars:

    >>> R_mars = N.zeros(u.shape, typecode=N.Float) + 3390e+3
    >>> R_mars[45,88] = 3390e+3 + 1000.0
    >>> curl = curl_2d(x, y, u, v, algorithm='spherepack', R_sphere=R_mars)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-1.21265e-07', '-6.068711e-08', '2.165924e-13', '6.068734e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['-1.150181e-20', '1.700353e-20', '1.227383e-20', '-2.283483e-20']
    >>> ['%.7g' % curl[-1,i] for i in range(10,14)]
    ['1.596643e-20', '5.265175e-21', '4.54957e-21', '-1.17195e-20']
    >>> ['%.7g' % curl[70,i] for i in range(130,134)]
    ['1.474248e-06', '1.470224e-06', '1.464409e-06', '1.456811e-06']

    (ii) Use order 1 spherical on Mars:

    >>> R_mars = N.zeros(u.shape, typecode=N.Float) + 3390e+3
    >>> R_mars[0,8] = 3390e+3 + 2000.0
    >>> curl = curl_2d(x,y,u,v,algorithm='order1_spherical',R_sphere=R_mars)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-1.224875e-07', '-6.128107e-08', '-9.383406e-23', '6.128107e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['1e+20', '1e+20', '1e+20', '1e+20']
    >>> ['%.7g' % curl[-1,i] for i in range(10,14)]
    ['1e+20', '1e+20', '1e+20', '1e+20']
    >>> ['%.7g' % curl[70,i] for i in range(130,134)]
    ['1.413254e-06', '1.408895e-06', '1.402819e-06', '1.395035e-06']
    """,


    'Additional Examples for Non-Scalar R_sphere:  Data Set 2':
    """
    (a) Import statements and create data:

    >>> import MA
    >>> import Numeric as N
    >>> from curl_2d import curl_2d
    >>> nx = 181
    >>> ny = 91
    >>> x = N.arange(nx) * 2.0 - 180.0
    >>> y = N.arange(ny) * 2.0 - 90.0
    >>> y_2d = N.reshape( N.repeat(y, nx), (ny, nx) )
    >>> x_2d = N.reshape( N.repeat(N.reshape(x,(1,nx)), ny), (ny, nx) )
    >>> u = N.sin(x_2d*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> v = N.sin((x_2d-30.)*N.pi/180.)*N.sin((y_2d-90.)*N.pi/30.)
    >>> N.putmask( u, N.where(N.absolute(u) < 1e-10, 1, 0), 0. )
    >>> N.putmask( v, N.where(N.absolute(v) < 1e-10, 1, 0), 0. )
    >>> N.putmask( u, N.where(N.absolute(y_2d) > 60., 1, 0), 0. )
    >>> N.putmask( v, N.where(N.absolute(y_2d) > 60., 1, 0), 0. )

    (b) Use order 1 Cartesian algorithm:  If x and y are in meters,
        and u and v are in m/s, the curl is in 1/s.  Note that the
        R_sphere keyword value has no effect with this algorithm:

    >>> R_earth = MA.zeros(u.shape, typecode=N.Float) + 6371220.0
    >>> R_earth[45,89] = 6371220.0 + 2000.0
    >>> curl = curl_2d( x, y, u, v, algorithm='order1_cartesian' \
                      , R_sphere=R_earth )
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-0.007251593', '-0.003628007', '0', '0.003628007']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['0', '0', '0', '0']

    (c) Use spectral method from SPHEREPACK (need to first remove the
        repeating dateline point from the domain).  Here we take x and
        y in deg, while u and v are in m/s.  If so the curl is in 1/s.
        These results as much less than for example (b) above since
        the domain is much larger.  We do two cases, one where the array
        is not all the same as the default Earth radius, and one where
        it is all the same as the default Earth radius:

    >>> x = x[0:-1]
    >>> u = u[:,0:-1]
    >>> v = v[:,0:-1]
    >>> R_earth = MA.zeros(u.shape, typecode=N.Float) + 6371220.0
    >>> R_earth[45,89] = 6371220.0 + 2000.0
    >>> curl = curl_2d(x, y, u, v, algorithm='spherepack', R_sphere=R_earth)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-6.436402e-08', '-3.219142e-08', '1.33852e-13', '3.220165e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['-2.299736e-20', '-4.806512e-21', '-9.283145e-22', '-1.368445e-20']

    >>> R_earth = MA.zeros(u.shape, typecode=N.Float) + 6371220.0
    >>> curl = curl_2d(x, y, u, v, algorithm='spherepack', R_sphere=R_earth)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-6.436402e-08', '-3.220152e-08', '1.33852e-13', '3.220165e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['-2.299736e-20', '-4.806512e-21', '-9.283145e-22', '-1.368445e-20']

    (d) Use order 1 spherical algorithm:  x and y are in deg and u and
        v are in m/s.  The curl is in 1/s.  Note that these results are
        nearly identical to those from SPHEREPACK:

    >>> R_earth = MA.zeros(u.shape, typecode=N.Float) + 6371220.0
    >>> R_earth[45,89] = 6371220.0 + 2000.0
    >>> curl = curl_2d( x, y, u, v, algorithm='order1_spherical' \
                      , R_sphere=R_earth )
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-6.517317e-08', '-3.259621e-08', '0', '3.260645e-08']
    >>> ['%.7g' % curl[0,i]  for i in range(8,12)]
    ['1e+20', '1e+20', '1e+20', '1e+20']

    (e) Use "default" algorithm, which for this domain chooses the 
        order 1 cartesian algorithm.  Thus, the R_sphere keyword
        setting should have no effect:

    >>> R_earth = MA.zeros(u.shape, typecode=N.Float) + 6371220.0
    >>> R_earth[45,89] = 6371220.0 + 2000.0
    >>> curl = curl_2d(x, y, u, v, algorithm='default', R_sphere=R_earth)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-0.007251593', '-0.003628007', '0', '0.003628007']

    (f) Use "default_spherical" algorithm with missing data (equal to
        1e+20), which because of the missing data will choose the order
        1 spherical algorithm.  We also do the calculation on Mars:

    >>> R_mars = N.zeros(u.shape, typecode=N.Float) + 3390e+3
    >>> R_mars[45,90] = 3390e+3 + 1500.0
    >>> R_mars[45,88] = 3390e+3 + 5000.0
    >>> u[30:46,90:114] = 1e+20
    >>> curl = curl_2d( x, y, u, v, algorithm='default_spherical' \
                      , R_sphere=R_mars)
    >>> ['%.7g' % curl[45,i] for i in range(88,92)]
    ['-1.223071e-07', '-6.128107e-08', '1e+20', '1e+20']
    """
    }



#- Execute doctest if module is run from command line:

if __name__ == "__main__":
    """Test the module.

    Tests the examples in all the module documentation strings, plus
    __test__.

    Note:  To help ensure that module testing of this file works, the
    parent directory to the current directory is added to sys.path.
    """
    import doctest, sys, os
    sys.path.append(os.pardir)
    doctest.testmod(sys.modules[__name__])




# ===== end file =====
