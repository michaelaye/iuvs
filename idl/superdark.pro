muv=iuvs_read_fits('/home/klay6683/superdark/mvn_iuv_l1b_periapse-orbit00437-muv_20141220T115005_v02_r01.fits.gz')

n_int=n_elements(muv.primary[0,0,*])
RESTORE,'/home/klay6683/superdark/xmas_aurora_superdark_v1.sav'
; IDL_SAVEFILE::RESTORE: Restored variable: xmas_AURORA_SUPERDARK.
; create calibration curve
avg_superdark=xmas_AURORA_SUPERDARK
n_spe = 256
n_spa = 7
senstivity_curve=muv.detector_dark_subtracted[*,3,9]/muv.primary[*,3,9]
dark1=muv.detector_dark[*,*,1]
x=reform(avg_superdark, n_spe*n_spa)
y=reform(dark1, n_spe*n_spa)
lin_fit=linfit(x,y,yfit=yfit)
superdark=lin_fit[0]+lin_fit[1]*avg_superdark
;plot,reform(superdark,n_spe*7),y,psym=3
;oplot,reform(superdark,n_spe*7),yfit,col='0000ff'x

primary_muv=muv.detector_raw-rebin(superdark,n_spe,n_spa,n_int)
primary_muv_cal=primary_muv/rebin(senstivity_curve,n_spe,n_spa,n_int)

muv=create_struct('primary',primary_muv_cal,remove_tags_f(muv,'primary'))
muv=create_struct('detector_dark_subtracted',primary_muv,remove_tags_f(muv,'detector_dark_subtracted'))

end