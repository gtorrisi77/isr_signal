# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 18:25:37 2024

@author: Gonzalo Torrisi
"""
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from scipy import integrate
from scipy.fft import fft, fftfreq
import plotly.graph_objs as go
from plotly.offline import plot

'cargo acelerograma'
reg=np.loadtxt('B-4.txt')


'calculos iniciales'
t=reg[:,0]
sig=reg[:,1]
dt=t[2]-t[1]
fsa=1/dt
g=9.81

'datos para filtrado'
f1=0.3
f2=17
orden=80
tmax=max(t)

'datos para correcion de linea base'
'orden polinomio inicial'
ndi=3
'orden polinomio luego del filtrado'
ndf=3

fig5=go.Figure()
plot(fig5)

'ploteo acelerograma original '
fig1=go.Figure(data=go.Scatter(x=t,y=sig,marker_color='blue',name='Original'))
fig1.update_layout(title="Original Accelerogram", titlefont_size=20,template='plotly_white',
    xaxis=dict(title='Time [s]',titlefont_size=16, tickfont_size=16, tickmode='linear',
        griddash='dash', dtick=2,gridcolor='grey',),
    yaxis=dict(
        title='Acceleration [g]', titlefont_size=16,tickfont_size=16, tickmode='linear',
        griddash='dash', dtick=0.25, gridcolor='grey', ),
    legend=dict( ),
    )
plot(fig1)





'espectro de fourier'
n=len(t)
f=np.zeros(n+1)
n2=int(n/2)+1
frec=np.zeros(n2)
amp=np.zeros(n2)

for i in range(1,n+1):
    f[i]=i/(dt*n)
for i in range(int(n/2)+1):
    frec[i]=f[i]

y1= fft(sig)
y=abs(y1)
ymax=max(y)
for i in range(int(n/2)+1):
    amp[i]=y[i]/ymax
plt.plot(frec,amp)
plt.title('fourier spectra')
plt.xlabel('freq [Hz]')
plt.ylabel('Amplitud')
plt.tight_layout()
plt.show()    



'ploteo ESPECTRO DE FOURIER'
fig2=go.Figure()
fig2.add_trace(go.Scatter(x=frec,y=amp,marker_color='red',name='Original'))
fig2.update_layout(title="Fourier Spectra", titlefont_size=20,template='plotly_white',
    xaxis=dict(title='frequency [Hz]',titlefont_size=16, tickfont_size=16, tickmode='linear',
        griddash='dash', dtick=2,gridcolor='grey',),
    yaxis=dict(
        title='Amplitude', titlefont_size=16,tickfont_size=16, tickmode='linear',
        griddash='dash', dtick=0.25, gridcolor='grey', ),
    legend=dict( ),
    )
plot(fig2)



'corrijo linea base inicial'

c11=np.polyfit(t,sig,ndi)
afit1=np.polyval(c11,t)
ac1=sig-afit1
vc1=integrate.cumtrapz(ac1, t, initial=0)
dc1=integrate.cumtrapz(vc1, t, initial=0)
'grafico'
plt.plot(t,sig,'grey',t, ac1,'g')
plt.title('Corrected Accelerogram')
plt.xlabel('Time [seconds]')
plt.ylabel('acceleration [g]')
plt.tight_layout()
plt.show()





'creo el filtro: orden "order" bandpass entre f1 y f2z'
sos= signal.butter(orden,[f1,f2], fs=fsa,btype='band',output='sos')
'aplico el filtro'
filtered = signal.sosfiltfilt(sos, ac1)
'grafico'
plt.plot(t,sig,'grey',t, filtered,'b')
plt.title('Filtered Accelerogram')
plt.xlabel('Time [seconds]')
plt.ylabel('acceleration [g]')
plt.tight_layout()
plt.show()

'velocidades y aceleraciones'
vel = integrate.cumtrapz(filtered, t, initial=0)
despl = integrate.cumtrapz(vel, t, initial=0)

'correccion de linea base inicial'
if ndf !=0:
    c1=np.polyfit(t,filtered,ndf)
    afit=np.polyval(c1,t)

    ac=filtered-afit
    vc=integrate.cumtrapz(ac, t, initial=0)
    dc=integrate.cumtrapz(vc, t, initial=0)
else:
    ac=filtered
    vc=vel
    dc=despl
    
    'GRAFICOS'
plt.plot(t,sig,'grey',t, filtered,'r')
plt.title('Corrected and filtered Accelerogram')
plt.xlabel('Time [seconds]')
plt.ylabel('acceleration [g]')
plt.tight_layout()
plt.show()   
    
plt.plot(t,vc*g,'b')
plt.title('Velocity')
plt.xlabel('Time [seconds]')
plt.ylabel('Velocity [m/s]')
plt.tight_layout()
plt.show()
plt.plot(t,dc*g,'b')
plt.title('Displacement')
plt.xlabel('Time [seconds]')
plt.ylabel('Displacement [m]')
plt.tight_layout()
plt.show()

'ploteo acelerograma '
fig3=go.Figure(data=go.Scatter(x=t,y=filtered,marker_color='red',name='filtered'))
fig3.add_trace(go.Scatter(x=t,y=sig, marker_color='grey', name='original'))
fig3.update_layout(title="Accelerogram", titlefont_size=20,template='plotly_white',
    xaxis=dict(title='Time [s]',titlefont_size=16, tickfont_size=16, tickmode='linear',
        griddash='dash', dtick=2,gridcolor='grey',),
    yaxis=dict(
        title='Acceleration [g]', titlefont_size=16,tickfont_size=16, tickmode='linear',
        griddash='dash', dtick=0.25, gridcolor='grey', ),
    legend=dict( ),
    )
plot(fig3)

