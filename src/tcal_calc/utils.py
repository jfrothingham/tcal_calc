


import numpy as np



from astropy import units as u
from astropy import constants as ac


def poly_pb(nu, coefs):
    """
    Equation
    """

    nu = nu.to(u.GHz).value

    # Calibrator flux density in Jy.
    snu = np.power(10., np.polyval(coefs, np.log10(nu)))*u.Jy

    return snu


def poly_ott(nu, coefs):
    """
    From Table 5 in Ott et al. 1994.
    """

    nu = nu.to(u.MHz).value
    lognu = np.log10(nu)

    snu = np.power(10., ( coefs[0] + coefs[1]*lognu + coefs[2]*np.power(lognu, 2.) ))*u.Jy

    return snu


def compute_sed(freq, scale, source, units='Jy'):
    """
    """

    coefs = cal_coefs[scale][source]
    #nu = freq.to(u.GHz).value

    # Calibrator flux density in Jy.
    snu = cal_coefs[scale]['method'](freq, coefs)
    #snu = np.power(10., np.polyval(coefs, np.log10(nu)))*u.Jy

    if 'K' in units:
        conv = jy2k(freq)
        snu *= conv

    return snu.to(units)


gbt = {'name': 'Green Bank Telescope',
       'diameter': 100*u.m,
       'surface rms': 350*u.um,
       'aperture efficiency': 0.71}

def jy2k(freq, eta_a_low_freq=gbt['aperture efficiency'],
         surf_rms=gbt['surface rms']):
    """
    Conversion factor between Jy and K.
    
    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency at which to evaluate the conversion factor.
    eta_a_low_freq : float, optional
        Low frequency aperture efficiency of the telescope.
    surf_rms : `~astropy.units.Quantity`, optional
        Surface root-mean-squared error.
        
    Returns
    -------
    jy2k : `~astropy.units.Quantity`
        Conversion factor in K/Jy.
    """
    
    lmbd = (ac.c/freq)
    eta_a = ruze(lmbd, eta_a_low_freq, surf_rms)
    # Specific gain: (2k/Ap)
    gain = 2.84
    
    return gain*eta_a*u.K/u.Jy


def ruze(lmbd, g0, surf_rms):
    """
    Ruze equation.
    
    Parameters
    ----------
    lmbd : float or `~astropy.units.quantity.Quantity`
        Wavelength.
    g0 : float
        Aperture efficiency at long wavelengths.
    surf_rms : float or `~astropy.units.quantity.Quantity`
        Surface rms.
    
    Returns
    -------
    eta_a : float
        Aperture efficiency at lmbd.
    """
    
    return g0*np.exp(-1.*np.power(4.*np.pi*surf_rms/lmbd, 2.))


calibrators = ['3C48', '3C123', '3C138', '3C147', '3C196', '3C286', '3C295', '3C348', '3C353', '3C380']


cal_coefs = {'Perley-Butler 2017':{'method' : poly_pb,
                                   '3C48' : [0.04980, -0.1914, -0.7553,  1.3253],
                                   '3C123': [0.00900, -0.0248, -0.1035, -0.7884,  1.8017],
                                   '3C138': [0.02230, -0.0102, -0.1552, -0.4981,  1.0088],
                                   '3C147': [0.02890, -0.0464,  0.0640, -0.2007, -0.6961, 1.4516],
                                   '3C196': [0.02010, -0.0200, -0.1534, -0.8530,  1.2872],
                                   '3C286': [0.03570, -0.1798, -0.4507,  1.2481],
                                   '3C295': [0.03990, -0.0347, -0.2780, -0.7658,  1.4701],
                                   '3C348': [0.00000, -0.0951, -1.0247,  1.8298],
                                   '3C353': [-0.0732, -0.0998, -0.6938,  1.8627],
                                   '3C380': [-0.1566, -0.1794,  0.0976,  0.0947, -0.7909, 1.2320],
                                   }, # Perley and Butler 2017
             'Ott 1994' : {'method': poly_ott,
                           '3C286': [0.956,  0.584, -0.1644],
                           '3C353': [3.148, -0.157, -0.0911],
                           }, # Ott et al. 1994
            }





def getApEff(elev, freq, coeffs=None):

    if coeffs == None:
        eff_long = 0.71

    eff_long=0.71
    rms = 230

    #theres some coefficient math here that is skipped in tcal_calc

    #where does 4.19... come from?
    arg = -(4.19169e-8 * rms * freq)**2

    return eff_long*np.exp( arg )

















#function getTau, freqs, coeffs=coeffs





def AirMass(el):
    if (el < 28):
        am = -0.023437  + 1.0140 / np.sin( (np.pi/180.)*(el + 5.1774 / (el + 3.3543) ) )
        print('A',am)
    else:
        am = 1./np.sin(np.pi*el/180.)
        print('B',am)
    return am
