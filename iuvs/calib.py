from numpy import poly1d
from collections import namedtuple

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


Coefficients = namedtuple('Coefficients', 'C0 C1 C2 C3 C4'.split())

def iuvs_dn_to_temp(dn, inverse=False, hw=False):
    """Convert DN to temperature or do the inverse.

    Parameters
    ----------
    dn : int
        Temperature in DN, or in degC if `inverse` is set to True
    inverse : bool, optional
        If set to True, the inverse calibration is performed.
        Default: False
    hw : bool, optional
        if set to True, use the thermistor polynomial, appropriate for
        all analog H/K TLM and case_temp science TLM. Default is use the
        detector on-chip conversion, appropriate to det_temp science TLM
        only.
    Returns
    -------
    float
        temperature in degC or DN, as appropriate to value of `inverse`
    """

    if hw:
        # ;All of the analog telemetry temperatures use the same split polynomial conversion:
        # ;DN from 0 to 8191: C0=1.8195   C1=-6.221E-3   C2=1.6259E-7   C3=-7.3832E-11   C4=-2.9933E-16
        # ;DN from 8192 to 16382: C0=450.47   C1=-6.5734E-2   C2=3.3093E-6   C3=-5.4216E-11   C4=-2.933E-16
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
                                    -2.9933E-16)
        poly0 = poly1d(*coeffs0_increasing_order.reverse())
        coeffs1_increasing_order = [450.47,
                                    -6.5734e-2,
                                    3.3093e-6,
                                    -5.4216e-11,
                                    -2.933e-16]
        poly1 = poly1d(*coeffs1_increasing_order.reverse())
