from pathlib import Path

import SpiceyPy as spice


def load_kernels():
    kernels_to_load = [
        'fk/maven_v06.tf',
        'ik/maven_iuvs_v01.ti',
        'ck/mvn_app_rel_131118_141010_v1.bc',
        'sclk/MVN_SCLKSCET.file.tsc',
        'spk/trj_c_131118-140923_rec_v1.bsp',
    ]
    spicedir = Path('/home/klay6683/IUVS-ITF-SW/anc/spice')
    mvnkerneldir = spicedir / 'mvn'
    lskdir = spicedir / 'generic_kernels/lsk'
    sclkdir = mvnkerneldir / 'sclk'
    for kernel in kernels_to_load:
        spice.furnsh(str(mvnkerneldir / kernel))
    for fname in sclkdir.iterdir():
        spice.furnsh(str(sclkdir / fname))
    spice.furnsh(str(lskdir / 'naif0011.tls'))
