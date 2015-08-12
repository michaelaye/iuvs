import numpy as np


# ;-----Original Message-----
# ;From: Justin Deighan
# ;Sent: Saturday, December 07, 2013 11:48 AM
# ;To: Chris Jeppesen
# ;Cc: Dale Theiling
# ;Subject: Temperature Conversion
# ;
# ;Hi Chris,
# ;
# ;I sat down with Dale yesterday and it turns out the "chip" temperatures that
# ;are in the science packets have a unique conversion from DNs. So that's why
# ;you were pulling temperatures from the FITS files that were different from
# ;what we saw during the checkout.
# ;
# ;Dale, could you pass that conversion on to Chris if you haven't done so
# ;already? Thanks.
# ;
# ;Justin
# ;  For the chip temperatures: (DN*-0.0749) + 174.08
# ;Everything else uses the split polynomials.
# ;
# ;D
# ;
# ;Dale Theiling
# ;email: dale.theiling@lasp.colorado.edu
# ;office: 303-492-5682


def iuvs_dn_to_temp(dn, inverse=False, det_temp=True):
    """Convert DN to temperature or do the inverse.

    Parameters
    ----------
    dn : int
        Temperature in DN, or in degC if `inverse` is set to True
    inverse : bool, optional
        If set to True, the inverse calibration is performed.
        Default: False
    det_temp : bool, optional
        if set to False, use the thermistor polynomial, appropriate for
        all analog H/K TLM and case_temp science TLM. Default is True, to
        use the detector on-chip conversion, appropriate to det_temp
        science TLM only.

    Returns
    -------
    float
        temperature in degC or DN, as appropriate to value of `inverse`
    """

    # convert input to float
    dn = np.array(dn, dtype='float')

    if not det_temp:
        # ;All of the analog telemetry temperatures use the same split polynomial conversion:
        # ;DN from 0 to 8191: C0=1.8195   C1=-6.221E-3   C2=1.6259E-7
        # ;C3=-7.3832E-11   C4=-2.9933E-16
        # ;DN from 8192 to 16382: C0=450.47   C1=-6.5734E-2   C2=3.3093E-6
        # ;C3=-5.4216E-11  C4=-2.933E-16
        # ;
        # ;D
        # ;
        # ;Dale Theiling
        # ;email: dale.theiling@lasp.colorado.edu
        # ;office: 303-492-5682

        coeffs0_increasing_order = [1.8195,
                                    -6.221E-3,
                                    1.6259E-7,
                                    -7.3832E-11,
                                    -2.9933E-16]
        poly0 = np.poly1d(list(reversed(coeffs0_increasing_order)))
        coeffs1_increasing_order = [450.47,
                                    -6.5734e-2,
                                    3.3093e-6,
                                    -5.4216e-11,
                                    -2.933e-16]
        poly1 = np.poly1d(list(reversed(coeffs1_increasing_order)))

        if not inverse:
            w = dn < 8192
            # create result array
            result = np.zeros_like(dn)
            # apply appropriate polynom
            result[w] = poly0(dn[w])
            result[~w] = poly1(dn[~w])
            return result
    else:  # if det_temp == True
        # for the chip temperatures: (DN * -0.0749) + 174.08
        a = -0.0749
        b = 174.08
        if not inverse:
            return a*dn + b
        else:
            T = dn
            return (T-b) / a


def convert_det_temp_to_C(dn, inverse=False):
    return iuvs_dn_to_temp(dn, inverse=inverse, det_temp=True)


def convert_case_temp_to_C(dn, inverse=False):
    return iuvs_dn_to_temp(dn, inverse=inverse, det_temp=False)
