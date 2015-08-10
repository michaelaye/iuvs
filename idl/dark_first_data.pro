pro dark_first_data
  file_array = file_search('/maven_iuvs/stage/products/level1b/mvn_iuv_l1b_outbound-orbit00113-mode1001-muv_20141019T120821_v00_r00.fits.gz', count=num_files)

  l1b=iuvs_read_fits(file_array[0])
  n_spe = n_elements(l1b.primary[*,0,0])
  n_spa = n_elements(l1b.primary[0,*,0])
  n_int = n_elements(l1b.primary[0,0,*])



  temp=dblarr(num_files)

  for k=0,num_files-1 do begin
    
    l1b=iuvs_read_fits(file_array[k])
    print, l1b.observation.product_id
    
    n_spe = n_elements(l1b.primary[*,0,0])
    n_spa = n_elements(l1b.primary[0,*,0])
    n_int = n_elements(l1b.primary[0,0,*])
    
    superlight = make_array(n_spe,n_spa,n_int,value=!values.f_nan)
    superdark = make_array(n_spe,n_spa,2,value=!values.f_nan)
    newdark=mean(superdark,dim=3)
    corrlight1=dblarr(n_spe,n_spa,n_int)

    for t=0,n_int-1 do begin
      if SIZE(l1b.detector_dark, /N_DIMENSIONS) GE 3 then begin
       
       superdark = reform(l1b.detector_dark)
       superlight[*,*,t] = reform(l1b.detector_raw[*,*,t])
       
      endif else begin
       
       superdark[*,*,t] = l1b.detector_dark[*,*]
       
      endelse
      
      
      if where(finite(superlight),/null) eq !NULL then continue
      
      ;indspadark = where(l1b.pixelgeometry[t].pixel_corner_mrh_alt[3,*] GE 200.)
      
      ;if n_elements(indspadark) GE 2 then begin
           
          X=findgen(n_spe)
          Y=dblarr(n_spe)
          Y=superlight[*,t]-superdark[*]
          ;for r=0,n_elements(indspadark)-1 do begin
           ;X[*,r]=findgen(n_spe)
    	   ;Y[*,r]=superlight[*,indspadark[r],t]- superdark[*,indspadark[r]]
    	   ;Y[*,r]=congrid(Y[where(Y[*,r] LE mean(Y[*,r])*2),r],n_elements(X[*,r]))
    	   ;stop
    	   ;result=linfit(X[*,r],Y[*,r])
            ;newdark[*,indspadark[r]]=(result(0)+X*result(1))+superdark[*,indspadark[r]]
            ;corrlight1[*,indspadark[r],t]=superlight[*,indspadark[r],t]-newdark
            
            Y=congrid(Y[where(Y LE mean(Y)*2)],n_spe)
            result=linfit(X,Y)
            newdark=(result(0)+X*result(1))+superdark
            corrlight1[*,t]=superlight[*,t]-newdark
         ; endfor
        
      ;endif
      ;stop
      plot,l1b.observation.wavelength[*,0],superlight[*,0,0]
     oplot,l1b.observation.wavelength[*,0],superdark[*,0,0]
     stop
    endfor
    

stop
       
        
        tv,bytscl(rebin(superlight[*,*,k+i*k],n_spe*2,n_spa*10,/sample),-3000.,3000.)

    plot, mean(median(superlight[*,*,*],dim=3),dim=2,/nan),ystyle=1,yrange=[0,1800.],title=l1b.engineering.det_temp
    oplot, mean(median(superdark[*,*,*],dim=3),dim=2,/nan), color='0000ff'x
    oplot, mean(median(superlight[*,*,*],dim=3),dim=2,/nan) - mean(median(superdark[*,*,*],dim=3),dim=2,/nan), color='00ff00'x
    stop
    ;oplot,X,result(0)+X*result(1),psym=4
    ;newdark=(result(0)+X*result(1))+(mean(median(superdark[*,*,*],dim=3),dim=2,/nan))
    ;corrlight1=mean(median(superlight[*,*,*],dim=3),dim=2,/nan)-newdark
    ;oplot,X,corrlight1

    ;corrlight1=congrid(corrlight1(where(corrlight1 LE mean(corrlight1)*5.)),n_elements(X))
    ;start = [0.,max(corrlight1),1.,1.]
    ;err = corrlight1-corrlight1+1

    ;result=MPFITFUN('MYSINUS',X,corrlight1,err,start)
    ;a=result[0]+result[1]*sin(result[2]*X+result[3])
    corrlight1(where(corrlight1 LE 0.))=0.
    tv,alog10(corrlight1)

    wait,1
  endfor
stop

end