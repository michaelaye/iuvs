;input
;  dn - Temperature in DN, or in degC if /inverse is set
;  /inverse - if set, invert the conversion: input is in degC, output is in DN
;  /hw - if set, use the thermistor polynomial, appropriate for all analog HK tml
;        and case_Temp science tlm. Default is use the detector on-chip conversion,
;        appropriate to det_temp science tlm only.
;output
;  temperature in degC or DN as appropriate
function iuvs_dn_to_temp,dn_,inverse=inv,hw=hw

;-----Original Message-----
;From: Justin Deighan 
;Sent: Saturday, December 07, 2013 11:48 AM
;To: Chris Jeppesen
;Cc: Dale Theiling
;Subject: Temperature Conversion
;
;Hi Chris,
;
;I sat down with Dale yesterday and it turns out the "chip" temperatures that 
;are in the science packets have a unique conversion from DNs. So that's why 
;you were pulling temperatures from the FITS files that were different from 
;what we saw during the checkout.
;
;Dale, could you pass that conversion on to Chris if you haven't done so 
;already? Thanks.
;
;Justin
;  For the chip temperatures: (DN*-0.0749) + 174.08
;Everything else uses the split polynomials.
;
;D
;
;Dale Theiling
;email: dale.theiling@lasp.colorado.edu
;office: 303-492-5682



  if keyword_set(hw) then begin
  ;All of the analog telemetry temperatures use the same split polynomial conversion:

  ;DN from 0 to 8191: C0=1.8195   C1=-6.221E-3   C2=1.6259E-7   C3=-7.3832E-11   C4=-2.9933E-16  
  ;DN from 8192 to 16382: C0=450.47   C1=-6.5734E-2   C2=3.3093E-6   C3=-5.4216E-11   C4=-2.933E-16
  ;
  ;D
  ;
  ;Dale Theiling
  ;email: dale.theiling@lasp.colorado.edu
  ;office: 303-492-5682
  
    poly0=[  1.8195d,-6.2210d-3,1.6259d-7,-7.3832d-11,-2.9933d-16]
    poly1=[450.47d,  -6.5734d-2,3.3093d-6,-5.4216d-11,-2.9330d-16]
    
    if ~keyword_set(inv) then begin
      result=dblarr(n_elements(dn_))
      w=where(dn_ lt 8192,count,comp=nw,ncomp=ncount)
      if  count gt 0 then result[ w]=poly(dn_[ w],poly0)
      if ncount gt 0 then result[nw]=poly(dn_[nw],poly1)
      return,result
    end else begin
      ;Technically there is a solution to the quartic equation, but that isn't really practical. 
      ;We'll use the same solution as in iuvs_volt_to_gain, bisection.
      ;
      ;Set up such that V is the independent variable and G is dependent
      T=double(DN_)
      DN=dblarr(n_elements(T))
      ;Set brackets on DN
      DNlo=DN
      DNhi=DN
      w=where(T le poly0[0],count,comp=nw,ncomp=ncount)
      if count gt 0 then begin
        DNlo[w]=0d
        DNhi[w]=8191d
      end
      if ncount gt 0 then begin
        DNlo[nw]=8192d
        DNhi[nw]=16382d
      end      
  
      w=where(T lt iuvs_dn_to_temp(8191),count)
      
      if count gt 0 then begin
        message,/info,"Input temperature out of range"
        DNlo[w]=8191
        DNhi[w]=8191
      end
      
      w=where(T gt iuvs_dn_to_temp(8192),count)
      if count gt 0 then begin
        message,/info,"Input temperature out of range"
        DNlo[w]=8192
        DNhi[w]=8192
      end
      for i=0,16 do begin 
        DNmid=(DNlo+DNhi)/2 
        Tmid=iuvs_dn_to_temp(DNmid)
        w=where(Tmid lt T,count,comp=nw,ncomp=ncount)
        if count gt 0 then DNhi[w]=DNmid[w]
        if ncount gt 0 then DNlo[nw]=DNmid[nw]
  ;      plot,[T,T],[dnlo,dnhi],/ynoz,/nodata
  ;      oplot,T,dnlo,color='0000ff'x
  ;      oplot,T,dnhi,color='ff0000'x
      end  
      return,fix(DNmid)
    end 
  end else begin
;  For the chip temperatures: (DN*-0.0749) + 174.08
    a=-0.0749d
    b=174.08d
    if ~keyword_set(inv) then begin
      return,a*dn_+b
    end else begin
      ;T=a*DN+b
      ;(T-b)=a*DN
      ;(T-b)/a=DN
      T=dn_
      return,(T-b)/a
    end
  end
end