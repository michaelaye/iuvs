;scale_dark.pro
;Nick Schneider, 6 June 14
;shows the need to scale down dark frame for optimal subtraction in a CruiseCal 2 frame

imstruct=iuvs_read_fits("~/MAVENData/CruiseCal2/mvn_iuv_l1b_cruisecal2-mode080-muv_20140521T120028_v00_r00.fits.gz")
dark=imstruct.DETECTOR_DARK
raw=imstruct.DETECTOR_RAW
rawa=avg(raw,2) 
!p.multi=0                                              

;show mismatch between light and dark values at unit scaling
window,0,xs=415,ys=400
for j=0,61 do begin
   plot,[0],[0],xrange=[0,341],yrange=[-1000,11000],/xsty,/ysty,/nodata,  $
      title=string(j)
   for i=0,10 do begin
      frac=0.8+i*0.03
      diff=rawa-frac*dark
      oplot,diff(*,j)+i*1000
   endfor
wait,0.05
endfor

;show decreased noise in averaged spectrum at ~0.9 dark scaling
window,1
plot,[0],[0],xrange=[0,341],yrange=[100,1700],/xsty,/ysty,/nodata,  $
   title=“average with different scaled darks (0.8-1.1)”
for i=0,10 do begin
   frac=0.8+i*0.03
   diff=rawa-frac*dark
   diffa=avg(diff,1)
   oplot,diffa+i*100
endfor

;display image subtractions for different scalings
window,2,xs=350,ys=870
for i=0,10 do begin
   frac=0.8+i*0.03
   diff=rawa-frac*dark
   tv,bytscl(diff,-100,1000),0,i*70
   xyouts,280,20+i*70,string(frac),/device,charsize=1.3
endfor
;dimply map of hottest pixels
tv,bytscl((rawa-(rawa<2000)),-100,3000),0,800

;plot dark current vs. exposure
window,4
hot_ave=fltarr(46)
warm_ave=fltarr(46)
hotpix=where(dark gt 2000)
warmpix=hotpix+1
dark_hot_ave=avg(dark(hotpix))
dark_warm_ave=avg(dark(warmpix))
raw_hot_ave=avg(rawa(hotpix))
raw_warm_ave=avg(rawa(warmpix))
light=raw_warm_ave-dark_warm_ave
raw_hot_dark_ave=raw_hot_ave-light
scale=raw_hot_dark_ave/raw_hot_ave
print,dark_hot_ave,dark_warm_ave,raw_hot_ave,raw_warm_ave,light,raw_hot_dark_ave,scale

for i=0,45 do begin
   rawi=raw(*,*,i)
   plot,raw(*,*,0),rawi,psym=2,symsize=.01,xr=[0,6000],yr=[0,6000],title=string(i)
   oplot,[0,6000],[0,6000]
   hot_ave(i)=avg(rawi(hotpix))
   warm_ave(i)=avg(rawi(warmpix))
   wait,0.05
endfor
plot,hot_ave,title='Dark Current in Cruisecal2-mode080-muv_20140521T120028',   $
    ytitle='Dark Current',xtitle='frame #',charsize=1.3
oplot,hot_ave-light,linestyle=4
oplot,warm_ave,linestyle=2
oplot,[0,46],[dark_hot_ave,dark_hot_ave],linestyle=3
xyouts,38,3200,'Raw data hot pixels',/data,charsize=1.3
xyouts,25,2500,'Raw data hot pixels - estimated light',/data,charsize=1.3
xyouts,2,2900,'Dark frame pixels',/data,charsize=1.3
xyouts,35,1000,'Raw data warm pixels',/data,charsize=1.3

;plot light vs dark values for different scalings
window,5
   for i=0,10 do begin
      frac=0.8+i*0.03
      diff=rawa-frac*dark
      plot,frac*dark,rawa,psym=1,symsize=0.01,title=string(frac),xr=[0,5000],yr=[0,5000]
      oplot,[0,5000],[0,5000]
      wait,1
   endfor

stop
end