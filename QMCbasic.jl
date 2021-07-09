using Jacobi
using DelimitedFiles
using Plots
using Interpolations
using Optim
pyplot()

cd(string(homedir(),"/Dropbox/PhD/QMC/basicQMC"))

#Units and Constants
const fm=1.0 #Main unit throughout code
const MeV=(1/197.33)*fm^-1
const mevpfm3=MeV/(fm^3) #To facilitate changing units
const fmm3=1/(fm^3)
# const n0=0.16fmm3 #Saturation density
const RB=0.8fm #Bag radius

#Step in density when calculating β equilibrium. Suggest between 0.1~0.01 n0
const dens_step=0.01fmm3

const n_points=9 #Number of points in momentum integral

#Baryon (bare) masses
const mn=939*MeV
const mp=mn
const mΛ=1107*MeV
const mΞ=1325*MeV

#Meson masses
const mσ=700MeV
const mω=783MeV
const mρ=770MeV
const mδ=983MeV

#Couplings 2007 ω=6.11; ρ=2.74; σ=10.03
const stdGsig=10.03fm^2
const stdgsig=sqrt(stdGsig*mσ^2)
const stdGomg=6.11fm^2
const stdgomg=sqrt(stdGomg*mω^2)
const stdGrho=2.74fm^2
const stdgrho=sqrt(stdGrho*mρ^2)
const stdGdel=0.0
const stdgdel=sqrt(stdGdel*mδ^2)

#Overlap
const λ3=0.02*fm^-1
const E0=0*fm^-4
const b=0.5*fm

#Lepton masses
const me=0.5*MeV
const mμ=105*MeV

######################################################################
######################################################################
#MAIN CODE
######################################################################
######################################################################

#Integration, Gauss-Legendre quadrature
const N=n_points

# Zeros of Legendre polinomials
lk=legendre_zeros(n_points) #The label "k" is always for integrating momentum

# Derivatives of legendre (see wiki on Gauss-Legendre quadrature)
dlk=dlegendre.(lk,length(lk))

# Gauss-Legendre quadrature Weights
wk=[2/((1-lk[i]^2)*(dlk[i])^2) for i in 1:length(dlk)]

# Legendre zeros go from -1 to 1. Need to dilate that interval to the correct integration range.
kpoint(t,max,min)=t*(max-min)/2+(max+min)/2 #This dilates [-1,1] linearly to [min,max]

function integrate1(F,min,max)
    #F must be a function of only k
    I=0.0
    for ik in 1:length(wk)
            Δ=(max-min)/2 #Jacobian of the kpoint.
            I=I+F(kpoint(lk[ik],min,max))*wk[ik]*Δ
    end
    return I
end
function integrate2(F,min,max)
    #F must be a function of (k1,k2)
    #Both min and max must be vectors e.g. min=[min_k1,min_k2]
    I=0.0
    for ik1 in 1:length(wk)
        for ik2 in 1:length(wk)
            Δ=(max[1]-min[1])/2*(max[2]-min[2])/2 #Jacobian of the kpoint transformations.
            I=I+F(kpoint(lk[ik1],min[1],max[1]),kpoint(lk[ik2],min[2],max[2]))*wk[ik1]*wk[ik2]*Δ
        end
    end
    return I
end

######################################################################
# Defining baryon effective masses

const rb=RB/fm #Unitless RB. As long as fm=1 this is moot, but in case we change units...

# #Here I'm assuming no delta meson interaction but you can use the polarizibilities from my other code below if you wish.
#Scalar polarizibilities from 2007 paper
const dσn=(0.0022+0.1055*rb-0.0178*rb^2)
const dσp=dσn
const dσΛ=(0.0016+0.0686*rb-0.0084*rb^2)
const dσΞ0=dσΞm=(-0.0014+0.0416*rb-0.0061*rb^2)
const dδn=dδp=dδΛ=dδΞ0=dδΞm=0.0
const d2p=d2n=d2Λ=d2Ξ0=d2Ξm=0.0

#Hyperon weights (linear multiplicative term on the mass)
const ωσn=ωσp=-1.0
const ωσΛ=-(0.6672+0.0462*rb-0.0021*rb^2)
const ωσΞm=ωσΞ0=-(0.3395+0.0282*rb-0.0128*rb^2)
const ωδn=ωδp=0.0
const ωδΛ=0.0
const ωδΞm=ωδΞ0=0.0

# #Scalar polatizibilities from my previous code for checking
#proton_fit = (-1.0020490905843504, -0.5000073592968122, 0.00039048040033750996, 0.001036167880136899, 0.00041629613837873477)
#(ωσp,ωδp,dσp,dδp,d2p)=proton_fit.*(1.,1.,1/MeV,1/MeV,1/MeV)
#
#neutron_fit = (-1.0020490905843504, 0.5000111670421129, 0.00039048040033750996, 0.001033289359647953, -0.0004166337075839663)
#(ωσn,ωδn,dσn,dδn,d2n)=neutron_fit.*(1.,1.,1/MeV,1/MeV,1/MeV)
#
#Λ_fit = (-0.703521253522257, 1.2692484847367558e-6, 0.00026586230236864375, 0.0006898190799256536, -1.1252306841256818e-7)
#(ωσΛ,ωδΛ,dσΛ,dδΛ,d2Λ)=Λ_fit.*(1.,1.,1/MeV,1/MeV,1/MeV)
#
#Ξ0_fit = (-0.34095286459095936, -0.5000086285452198, 0.0001413602490885336, 0.0003463488002066139, 0.0004164086614463546)
#(ωσΞ0,ωδΞ0,dσΞ0,dδΞ0,d2Ξ0)=Ξ0_fit.*(1.,1.,1/MeV,1/MeV,1/MeV)
#
#Ξm_fit = (-0.34095286459095936, 0.5000098977936963, 0.0001413602490885336, 0.0003434702797183014, -0.0004165211845146568)
#(ωσΞm,ωδΞm,dσΞm,dδΞm,d2Ξm)=Ξm_fit.*(1.,1.,1/MeV,1/MeV,1/MeV)

Mn(sig,del,gsig,gdel)= mn +ωσn*gsig*sig +(dσn)*(gsig*sig)^2 +ωδn*gdel*del +(dδn)*(gdel*del)^2 + (d2n)*gdel*del*gsig*sig
Mp(sig,del,gsig,gdel)= mp +ωσp*gsig*sig +(dσp)*(gsig*sig)^2 +ωδp*gdel*del +(dδp)*(gdel*del)^2 + (d2p)*gdel*del*gsig*sig
MΛ(sig,del,gsig,gdel)= mΛ +ωσΛ*gsig*sig +(dσΛ)*(gsig*sig)^2 +ωδΛ*gdel*del +(dδΛ)*(gdel*del)^2 + (d2Λ)*gdel*del*gsig*sig
MΞ0(sig,del,gsig,gdel)= mΞ +ωσΞ0*gsig*sig +(dσΞ0)*(gsig*sig)^2 +ωδΞ0*gdel*del +(dδΞ0)*(gdel*del)^2 + (d2Ξ0)*gdel*del*gsig*sig
MΞm(sig,del,gsig,gdel)= mΞ +ωσΞm*gsig*sig +(dσΞm)*(gsig*sig)^2 +ωδΞm*gdel*del +(dδΞm)*(gdel*del)^2 + (d2Ξm)*gdel*del*gsig*sig

######################################################################
# Defining mean field functions

kf(n)=(3π^2*n)^(1/3) #Fermi momentum as func of density n

const ωωn=1.0 #omega weights, see eq (3) of 2007 paper
const ωωp=1.0
const ωωΛ=1.0-1/3
const ωωΞ=1.0-2/3
function ω(nn,np,nΛ,nΞ0,nΞm,gomg)
    meanfield=0.
    for X in [nn*ωωn,np*ωωp,nΛ*ωωΛ,nΞ0*ωωΞ,nΞm*ωωΞ]
        meanfield=meanfield +gomg*X/(mω^2)
    end
    return meanfield
end

const In=-1/2 #Isospin, rho weights
const Ip=1/2
const IΛ=0.0
const IΞ0=1/2
const IΞm=-1/2
function ρ(nn,np,nΛ,nΞ0,nΞm,grho)
    meanfield=0.
    for X in [nn*In,np*Ip,nΛ*IΛ,nΞ0*IΞ0,nΞm*IΞm]
        meanfield=meanfield +grho*X/(mρ^2)
    end
    return meanfield
end

function quickNR(f,d,s,er;debug=false,absit=false) #Quick code of Newton-Rapshon
    df(x)=(f(x+d)-f(x))/d
    x0=s
    erro=10er
    while erro>er
        x1=(x0-f(x0)/df(x0))
        if absit
            x1=abs(x0-f(x0)/df(x0))
        end
        erro=abs(x0-x1)
        x0=x1
        if debug
            @show erro
        end
    end
    return x0
end

######################################################################
# Self Consistency Equations (SCE)

# Vertex Functions, i.e., derivatives of the dynamical couplings
Γσn(sig,del,gsig,gdel)=-(ωσn*gsig +2dσn*gsig^2*sig +d2n*gdel*del*gsig)
Γδn(sig,del,gsig,gdel)=-(ωδn*gdel +2dδn*gdel^2*del +d2n*gdel*sig*gsig)
gΓδn(sig,del,gsig,gdel)=(ωδn*gdel +d2n*gdel*gsig*sig)
Γσp(sig,del,gsig,gdel)=-(ωσp*gsig +2dσp*gsig^2*sig +d2p*gdel*del*gsig)
Γδp(sig,del,gsig,gdel)=-(ωδp*gdel +2dδp*gdel^2*del +d2p*gdel*sig*gsig)
gΓδp(sig,del,gsig,gdel)=(ωδp*gdel +d2p*gdel*gsig*sig)
ΓσΛ(sig,del,gsig,gdel)=-(ωσΛ*gsig +2dσΛ*gsig^2*sig +d2Λ*gdel*del*gsig)
ΓδΛ(sig,del,gsig,gdel)=-(ωδΛ*gdel +2dδΛ*gdel^2*del +d2Λ*gdel*sig*gsig)
gΓδΛ(sig,del,gsig,gdel)=(ωδΛ*gdel +d2Λ*gdel*gsig*sig)
ΓσΞ0(sig,del,gsig,gdel)=-(ωσΞ0*gsig +2dσΞ0*gsig^2*sig +d2Ξ0*gdel*del*gsig)
ΓδΞ0(sig,del,gsig,gdel)=-(ωδΞ0*gdel +2dδΞ0*gdel^2*del +d2Ξ0*gdel*sig*gsig)
gΓδΞ0(sig,del,gsig,gdel)=(ωδΞ0*gdel +d2Ξ0*gdel*gsig*sig)
ΓσΞm(sig,del,gsig,gdel)=-(ωσΞm*gsig +2dσΞm*gsig^2*sig +d2Ξm*gdel*del*gsig)
ΓδΞm(sig,del,gsig,gdel)=-(ωδΞm*gdel +2dδΞm*gdel^2*del +d2Ξm*gdel*sig*gsig)
gΓδΞm(sig,del,gsig,gdel)=(ωδΞm*gdel +d2Ξm*gdel*gsig*sig)

# Self Consistency Equations
function SCEσ(sig,del,nn,np,nΛ,nΞ0,nΞm,gsig,gdel)
    cte=2/((2π)^3)

    ins(m)=k->(4π)*k^2*m/sqrt(k^2+m^2) #scalar density "ns" integrand

    #neutron contribution
    p1=Γσn(sig,del,gsig,gdel)*integrate1(ins(Mn(sig,del,gsig,gdel)),0.0,kf(nn)+1e-10*MeV)

    #proton contribution
    p1=p1+Γσp(sig,del,gsig,gdel)*integrate1(ins(Mp(sig,del,gsig,gdel)),0.0,kf(np)+1e-10*MeV)

    #Λ contribution
    p1=p1+ΓσΛ(sig,del,gsig,gdel)*integrate1(ins(MΛ(sig,del,gsig,gdel)),0.0,kf(nΛ)+1e-10*MeV)

    #Ξ0 contribution
    p1=p1+ΓσΞ0(sig,del,gsig,gdel)*integrate1(ins(MΞ0(sig,del,gsig,gdel)),0.0,kf(nΞ0)+1e-10*MeV)

    #Ξm contribution
    p1=p1+ΓσΞm(sig,del,gsig,gdel)*integrate1(ins(MΞm(sig,del,gsig,gdel)),0.0,kf(nΞm)+1e-10*MeV)
    
    return (mσ^2*sig + (λ3/2)*gsig^3*sig^2 - p1*cte)
end
function SCEδ(sig,del,nn,np,nΛ,nΞ0,nΞm,gsig,gdel)
    cte=2/((2π)^3) /mδ^2

    ins(m)=k->(4π)*k^2*m/sqrt(k^2+m^2) #scalar density "ns" integrand

    #neutron contribution
    p1=Γδn(sig,del,gsig,gdel)*integrate1(ins(Mn(sig,del,gsig,gdel)),0.0,kf(nn)+1e-10*MeV)

    #proton contribution
    p1=p1+Γδp(sig,del,gsig,gdel)*integrate1(ins(Mp(sig,del,gsig,gdel)),0.0,kf(np)+1e-10*MeV)

    #Λ contribution
    p1=p1+ΓδΛ(sig,del,gsig,gdel)*integrate1(ins(MΛ(sig,del,gsig,gdel)),0.0,kf(nΛ)+1e-10*MeV)

    #Ξ0 contribution
    p1=p1+ΓδΞ0(sig,del,gsig,gdel)*integrate1(ins(MΞ0(sig,del,gsig,gdel)),0.0,kf(nΞ0)+1e-10*MeV)

    #Ξm contribution
    p1=p1+ΓδΞm(sig,del,gsig,gdel)*integrate1(ins(MΞm(sig,del,gsig,gdel)),0.0,kf(nΞm)+1e-10*MeV)

    return (-del + p1*cte)
end

function σ0(del,nn,np,nΛ,nΞ0,nΞm,gsig,gdel,er=0.1MeV,d=0.1MeV)
    f1(sig)=SCEσ(sig,del,nn,np,nΛ,nΞ0,nΞm,gsig,gdel)
    return quickNR(f1,d,10MeV,er)#,absit=true)
end
function δ0(sig,nn,np,nΛ,nΞ0,nΞm,gsig,gdel,er=0.1MeV,d=0.1MeV)
    f1(del)=SCEδ(sig,del,nn,np,nΛ,nΞ0,nΞm,gsig,gdel)
    return quickNR(f1,d,10MeV,er)
end
function σδ(nn,np,nΛ,nΞ0,nΞm,gsig,gdel,er=0.1MeV)
    sig0=10MeV
    del0=2MeV
    erro=10er
    while erro>er
        sig1=σ0(del0,nn,np,nΛ,nΞ0,nΞm,gsig,gdel)
        del1=δ0(sig1,nn,np,nΛ,nΞ0,nΞm,gsig,gdel)
        erro=sqrt((sig0-sig1)^2 + (del0-del1)^2)
        sig0=sig1
        del0=del1
    end
    return sig0,del0
end

######################################################################
# Defining averages of the hamiltonian, aka energy density

###
#Free bit

#Analytic integral of x^2*sqrt(x^2+m^2) from 0 to k
integ(k,mf)=(1/8)*( k*sqrt(mf^2+k^2)*(mf^2+2*k^2)-mf^4*log(sqrt(mf^2+k^2)+k) + mf^4*log(sqrt(mf^2)) )

function H0(nn,np,nΛ,nΞ0,nΞm,ne,nμ,gsig,gdel)
    cte=(2/((2π)^3))*(4π)
    (S,D)=σδ(nn,np,nΛ,nΞ0,nΞm,gsig,gdel) #Calculate sigma&delta bar only once

    #neutron
    h0=integ(kf(nn),Mn(S,D,gsig,gdel))
    #proton
    h0=h0+integ(kf(np),Mp(S,D,gsig,gdel))
    #Lambda
    h0=h0+integ(kf(nΛ),MΛ(S,D,gsig,gdel))
    #Ξs
    h0=h0+integ(kf(nΞ0),MΞ0(S,D,gsig,gdel))
    h0=h0+integ(kf(nΞm),MΞm(S,D,gsig,gdel))
    #Leptons
    h0=h0+integ(kf(ne),me)
    h0=h0+integ(kf(nμ),mμ)

    return h0*cte
end

###
#Fock terms
function Fock(nn,np,nΛ,nΞ0,nΞm,gsig,gdel,gomg,grho) #Minifock functions defined below!!!!
    (S,D)=σδ(nn,np,nΛ,nΞ0,nΞm,gsig,gdel)

    #neutron
    result=minifocksigma(Mn,Γσn,S,D,nn,gsig,gdel)
    result=result+minifockomega(ωωn,nn,gomg)

    #proton
    result=result+minifocksigma(Mp,Γσp,S,D,np,gsig,gdel)
    result=result+minifockomega(ωωp,np,gomg)

    #rho exchange
    result=result+minifockrho(nn,nn,grho)*(1/2)^2
    result=result+minifockrho(nn,np,grho)
    result=result+minifockrho(np,np,grho)*(1/2)^2
    #delta exchange
    result=result+minifockdelta(Mn,Mn,Γδn,Γδn,S,D,nn,nn,gsig,gdel)*(1/2)^2
    result=result+minifockdelta(Mn,Mp,gΓδn,gΓδp,S,D,nn,np,gsig,gdel)
    result=result+minifockdelta(Mp,Mp,Γδp,Γδp,S,D,np,np,gsig,gdel)*(1/2)^2

    #lambda
    result=result+minifocksigma(MΛ,ΓσΛ,S,D,nΛ,gsig,gdel)
    result=result+minifockomega(ωωΛ,nΛ,gomg)

    #Ξ0
    result=result+minifocksigma(MΞ0,ΓσΞ0,S,D,nΞ0,gsig,gdel)
    result=result+minifockomega(ωωΞ,nΞ0,gomg)

    #Ξm
    result=result+minifocksigma(MΞm,ΓσΞm,S,D,nΞm,gsig,gdel)
    result=result+minifockomega(ωωΞ,nΞm,gomg)

    #Ξ rho exchange
    result=result+minifockrho(nΞ0,nΞ0,grho)*(1/2)^2
    result=result+minifockrho(nΞ0,nΞm,grho)
    result=result+minifockrho(nΞm,nΞm,grho)*(1/2)^2
    #Ξ delta exchange
    result=result+minifockdelta(MΞ0,MΞ0,ΓδΞ0,ΓδΞ0,S,D,nΞ0,nΞ0,gsig,gdel)*(1/2)^2
    result=result+minifockdelta(MΞ0,MΞm,gΓδΞ0,gΓδΞm,S,D,nΞ0,nΞm,gsig,gdel)
    result=result+minifockdelta(MΞm,MΞm,ΓδΞm,ΓδΞm,S,D,nΞm,nΞm,gsig,gdel)*(1/2)^2
    return result
end

#Meson propagator integrated in θ
meson(k1,k2,m)=-(4π*2π)*(1/(2k1*k2))*log((k1^2+k2^2+m^2-2k1*k2)/(k1^2+k2^2+m^2+2k1*k2))

function minifocksigma(MZ,Γ,S,D,nZ,gsig,gdel)
    c=+(Γ(S,D,gsig,gdel))^2*1/((2π)^6)
    integrand(k1,k2)=k1^2*k2^2*meson(k1,k2,mσ)*MZ(S,D,gsig,gdel)^2/sqrt((k1^2+MZ(S,D,gsig,gdel)^2)*(k2^2+MZ(S,D,gsig,gdel)^2))
    return integrate2(integrand,[0.0,0.0],[kf(nZ)+1e-10*MeV,kf(nZ)+1e-10*MeV])*c
end
function minifockdelta(MZ1,MZ2,Γ1,Γ2,S,D,nZ1,nZ2,gsig,gdel)
    c=+(Γ1(S,D,gsig,gdel)*Γ2(S,D,gsig,gdel))*1/((2π)^6)
    integrand(k1,k2)=k1^2*k2^2*meson(k1,k2,mδ)*MZ1(S,D,gsig,gdel)*MZ2(S,D,gsig,gdel)/sqrt((k1^2+MZ1(S,D,gsig,gdel)^2)*(k2^2+MZ2(S,D,gsig,gdel)^2))
    return integrate2(integrand,[0.0,0.0],[kf(nZ1)+1e-10*MeV,kf(nZ2)+1e-10*MeV])*c
end
function minifockomega(ωωZ,nZ,gomg)
    c=-(ωωZ*gomg)^2*1/((2π)^6)
    integrand(k1,k2)=k1^2*k2^2*meson(k1,k2,mω)
    return integrate2(integrand,[0.0,0.0],[kf(nZ)+1e-10*MeV,kf(nZ)+1e-10*MeV])*c
end
function minifockrho(nZ1,nZ2,grho)
    c=-(grho)^2*1/((2π)^6)
    integrand(k1,k2)=k1^2*k2^2*meson(k1,k2,mρ)
    return integrate2(integrand,[0.0,0.0],[kf(nZ1)+1e-10*MeV,kf(nZ2)+1e-10*MeV])*c
end

###
#Pions

const gA=1.26
const fπ=93MeV
const mπ=139MeV

function Fockπ(nn,np,nΛ,nΞ0,nΞm)
    res=minifockpi(np,np)
    res=res+minifockpi(np,nn)*4.
    res=res+minifockpi(nn,nn)
    res=res+minifockpi(nΞm,nΞm)*(1/25)
    res=res+minifockpi(nΞm,nΞ0)*(4/25)
    res=res+minifockpi(nΞ0,nΞ0)*(1/25)
    return res
end

function minifockpi(nZ1,nZ2)
    c=(gA/(2fπ))^2*1/((2π)^6)
    Jintegrand(k1,k2)=-mπ^2*k1^2*k2^2*meson(k1,k2,mπ)
    return integrate2(Jintegrand,[0.0,0.0],[kf(nZ1)+1e-10*MeV,kf(nZ2)+1e-10*MeV])*c
end

function overlap(nn,np,nΛ,nΞ0,nΞm)
    rbar=(nn+np+nΛ+nΞ0+nΞm)^(-1/3)
    return overlap=E0*exp(-rbar^2/b^2)
end

function H(nn,np,nΛ,nΞ0,nΞm,ne,nμ,gsig,gdel,gomg,grho)
    (S,D)=σδ(nn,np,nΛ,nΞ0,nΞm,gsig,gdel)
    MeanFields=mσ^2*S^2/2 + (λ3/6)*(gsig*S)^3 + mδ^2*D^2/2 + mρ^2*ρ(nn,np,nΛ,nΞ0,nΞm,grho)^2/2 + mω^2*ω(nn,np,nΛ,nΞ0,nΞm,gomg)^2/2
    return H0(nn,np,nΛ,nΞ0,nΞm,ne,nμ,gsig,gdel)+Fock(nn,np,nΛ,nΞ0,nΞm,gsig,gdel,gomg,grho)+Fockπ(nn,np,nΛ,nΞ0,nΞm)+MeanFields+overlap(nn,np,nΛ,nΞ0,nΞm)
end

#testing and timing
Fock(0.16*fmm3,2*0.16*fmm3,0.1*0.16*fmm3,0,0,stdgsig,stdgdel,stdgomg,stdgrho)/mevpfm3
Fockπ(0.16*fmm3,2*0.16*fmm3,0.1*0.16*fmm3,0,0)/mevpfm3
overlap(0.16*fmm3,2*0.16*fmm3,0.1*0.16*fmm3,0,0)/mevpfm3
@time @show ρ(0.16*fmm3*5,0.16*fmm3,0,0,0,stdgrho)/MeV
@time @show ω(0.16*fmm3/2,0.16*fmm3/2,0,0,0,stdgomg)/MeV
@time @show σδ(0.16*fmm3/2,0.16*fmm3/2,0,0,0,stdgsig,stdgdel)./MeV
@time @show σδ(5*0.16*fmm3,0.16*fmm3,0,0,0,stdgsig,stdgdel)./MeV
@time @show σδ(0.16*fmm3,5*0.16*fmm3,0,0,0,stdgsig,stdgdel)./MeV
H(0.16*fmm3/2,0.16*fmm3/2,0,0,0,0,0,stdgsig,stdgdel,stdgomg,stdgrho)/mevpfm3

nrange=collect(0.0:dens_step:1.0fmm3)
@time htemp=H.(nrange/2,nrange/2,0,0,0,0,0,stdgsig,stdgdel,stdgomg,stdgrho)./mevpfm3
plot(nrange,htemp)

function d1f(f,n,del) #Higher precision derivative
    if n-2*del>0
        return (-f(n+2*del)+8*f(n+del)-8*f(n-del)+f(n-2*del))/(12*del)
    end
    return (f(n+del)-f(n))/del
end

function NMparameters(gsig,gdel,gomg,grho)
    ℰfit(n)=H(n/2,n/2,0,0,0,0,0,gsig,gdel,gomg,grho)/n-mn
    dEfit(x)=d1f(ℰfit,x,0.001fmm3)
    Sfit(n)=(H(n,0,0,0,0,0,0,gsig,gdel,gomg,grho)-H(n/2,n/2,0,0,0,0,0,gsig,gdel,gomg,grho))/n
    nsat=quickNR(dEfit,0.001fmm3,0.01fmm3,0.001fmm3,absit=true)
    return nsat,ℰfit(nsat),Sfit(nsat)
end
function NMparametersFull(gsig,gdel,gomg,grho)
    ℰfit(n)=H(n/2,n/2,0,0,0,0,0,gsig,gdel,gomg,grho)/n-mn
    dEfit(x)=d1f(ℰfit,x,0.001fmm3)
    Sfit(n)=(H(n,0,0,0,0,0,0,gsig,gdel,gomg,grho)-H(n/2,n/2,0,0,0,0,0,gsig,gdel,gomg,grho))/n
    nsat=quickNR(dEfit,0.001fmm3,0.01fmm3,0.001fmm3,absit=true)
    function K0(n,del=0.003*fm^-3)
        return 9*n^2*(d1f(dEfit,n,del))
    end
    function L0(n,del=0.003*fm^-3)
        de1st(n)=d1f(Sfit,n,del)
        return 3*n*(de1st(n))
    end
    return nsat,ℰfit(nsat),Sfit(nsat),K0(nsat),L0(nsat)
end
#Check current NM parameters
@time NMparameters(stdgsig,stdgdel,stdgomg,stdgrho)./(fmm3,MeV,MeV)

#Fit function without delta
function NMfit_noδ(P) #P=[dens,bind,satener]
    g(X)=(sum(( (NMparameters(X[1],0,X[2],X[3]) .-P)./P ).^2 ))
    res=optimize(g,[stdgsig,stdgomg,stdgrho],method=NelderMead(),g_tol=1e-7)
    @show res
    return res.minimizer[1],0.0,res.minimizer[2],res.minimizer[3]
end

@time fitresult=NMfit_noδ([0.148fmm3,-15.8MeV,30MeV])

@show fitresult.^2 ./(mσ^2,mδ^2,mω^2,mρ^2) ./(fm^2)
@show NMparametersFull(fitresult...)./(fmm3,MeV,MeV,MeV,MeV)

@show stdgsig,stdgdel,stdgomg,stdgrho

# #You can setup the couplings by hand w/ the line below
# fitresult=sqrt.([9.5,0.0,5.67,2.35].*[mσ^2,mδ^2,mω^2,mρ^2])
# fitresult=sqrt.([11.25,0.0,7.1,3.5].*[mσ^2,mδ^2,mω^2,mρ^2])

function NMfitδ(Gdelta,P) #P=[dens,bind,satener]
    gdelta=sqrt(mδ^2*Gdelta)
    g(X)=sum( abs.((NMparameters(X[1],gdelta,X[2],X[3]).-P)./P) )
    res=optimize(g,[stdgsig,gdelta,stdgomg,stdgrho],NelderMead())
    @show res
    return res.minimizer[1],gdelta,res.minimizer[2],res.minimizer[3]
end

Fock(0.17*fmm3,2*0.17*fmm3,0.1*0.17*fmm3,0,0,fitresult...)/mevpfm3
Fockπ(0.17*fmm3,2*0.17*fmm3,0.1*0.17*fmm3,0,0)/mevpfm3
@time @show ρ(0.17*fmm3*5,0.17*fmm3,0,0,0,stdgrho)/MeV
@time @show ω(0.17*fmm3/2,0.17*fmm3/2,0,0,0,stdgomg)/MeV
@time @show σδ(0.17*fmm3/2,0.17*fmm3/2,0,0,0,stdgsig,stdgdel)./MeV
@time @show σδ(5*0.17*fmm3,0.17*fmm3,0,0,0,stdgsig,stdgdel)./MeV
@time @show σδ(0.17*fmm3,5*0.17*fmm3,0,0,0,stdgsig,stdgdel)./MeV
H(0.17*fmm3/2,0.17*fmm3/2,0,0,0,0,0,fitresult...)/mevpfm3


######################################################################
# Defining Chemical Potential and Pressure

function chempot(Index,N)
    (nn,np,nΛ,nΞ0,nΞm,ne,nμ)=N
    function l(x)
        d=zeros(length(N))
        d[Index]=x-N[Index]
        return d
    end
    func=x->H( (N.+l(x))...,fitresult...)
    return d1f(func,N[Index],0.001fmm3)
end

μn(nn,np,nΛ,nΞ0,nΞm,ne,nμ)=chempot(1,[nn,np,nΛ,nΞ0,nΞm,ne,nμ])
μp(nn,np,nΛ,nΞ0,nΞm,ne,nμ)=chempot(2,[nn,np,nΛ,nΞ0,nΞm,ne,nμ])
μΛ(nn,np,nΛ,nΞ0,nΞm,ne,nμ)=chempot(3,[nn,np,nΛ,nΞ0,nΞm,ne,nμ])
μΞ0(nn,np,nΛ,nΞ0,nΞm,ne,nμ)=chempot(4,[nn,np,nΛ,nΞ0,nΞm,ne,nμ])
μΞm(nn,np,nΛ,nΞ0,nΞm,ne,nμ)=chempot(5,[nn,np,nΛ,nΞ0,nΞm,ne,nμ])
μe(nn,np,nΛ,nΞ0,nΞm,ne,nμ)=chempot(6,[nn,np,nΛ,nΞ0,nΞm,ne,nμ])
μμ(nn,np,nΛ,nΞ0,nΞm,ne,nμ)=chempot(7,[nn,np,nΛ,nΞ0,nΞm,ne,nμ])

function P(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=nn*μn(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=p+np*μp(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=p+nΛ*μΛ(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=p+nΞ0*μΞ0(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=p+nΞm*μΞm(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=p+ne*μe(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=p+nμ*μμ(nn,np,nΛ,nΞ0,nΞm,ne,nμ)
    p=p-H(nn,np,nΛ,nΞ0,nΞm,ne,nμ,fitresult...)
    return p
end

######################################################################
#Species Fraction, minimizing the energy density in β-equilibrium

function cut(x,c=1e-4fmm3)
    if x<c
        return 0.0
    end
    return x
end

function β(nb,x0=zeros(4).+0.01fmm3)
    function eliminated(nb2,ne,nΛ,nΞ0,nΞm)
        (ne,nΛ,nΞ0,nΞm)=cut.([ne,nΛ,nΞ0,nΞm])
        nμ=real(sqrt(kf(ne)^2+me^2-mμ^2 +0.0*im))^3 /(3π^2)
        np=ne+nμ+nΞm
        nn=cut(nb2-(np+nΛ+nΞ0+nΞm))
        return H(nn,np,nΛ,nΞ0,nΞm,ne,nμ,fitresult...)
    end
    velim(N)=eliminated(nb,N...)
    opt=optimize(velim,x0,NelderMead(),Optim.Options(g_tol=1e-8))
    (βne,βnΛ,βnΞ0,βnΞm)=cut.(opt.minimizer[:])
    βnμ=real(sqrt(kf(βne)^2+me^2-mμ^2 +0.0*im))^3 /(3π^2)
    βnp=βne+βnμ+βnΞm
    βnn=cut(nb-(βnp+βnΛ+βnΞ0+βnΞm))
    return [βnn,βnp,βnΛ,βnΞ0,βnΞm,βne,βnμ],[βne,βnΛ,βnΞ0,βnΞm],velim([βne,βnΛ,βnΞ0,βnΞm])
end

function βnohyp(nb,x0=zeros(1).+0.01fmm3)
    function eliminated(nb2,ne)
        (ne)=cut(ne)
        nμ=real(sqrt(kf(ne)^2+me^2-mμ^2 +0.0*im))^3 /(3π^2)
        np=ne+nμ
        nn=cut(nb2-(np))
        return H(nn,np,0.0,0.0,0.0,ne,nμ,fitresult...)
    end
    velim(N)=eliminated(nb,N...)
    opt=optimize(velim,zeros(1).+0.01fmm3,NelderMead(),Optim.Options(g_tol=1e-8))
    (βne)=cut(opt.minimizer[1])
    βnμ=real(sqrt(kf(βne)^2+me^2-mμ^2 +0.0*im))^3 /(3π^2)
    βnp=βne+βnμ
    βnn=cut(nb-(βnp))
    return [βnn,βnp,0.0,0.0,0.0,βne,βnμ],[βne],velim([βne])
end

function SF(__β__)
    nvar=4
    if __β__==βnohyp
        nvar=1
    end
    beta=zeros(length(nrange),7)
    E=zeros(length(nrange))
    Pv=zeros(length(nrange))
    x0=zeros(nvar).+0.01fmm3
    println("Calculating β-equilibrium")
    for i in 1:length(nrange)
        nbi=nrange[i]
        (beta[i,:],x0[:],E[i])=__β__(nbi,x0)
        Pv[i]=P(beta[i,:]...)
    end
    print("Done:")
    return beta,E,Pv
end

@time Btab,Etab,Ptab=SF(β)

######################################################################
# Plotting and Exporting

#Species Fraction
plot()
labels=["n","p","Λ","Ξ⁰","Ξ⁻","e","μ"]
for i in 1:7
    plot!(nrange,Btab[:,i]./nrange,label=labels[i],yaxis=:log)
end
plot!(xaxis=[0,1],yaxis=[1e-3,1e0],xlabel="n\$_b\$",ylabel="Density Fraction",guidefontsize=8,title_location=:left,xticks=0:0.1:2,frame=:box,size=(600,300))
plot!(legendfontsize=6,legend=:outertopright)
# savefig("speciesfrac.pdf")

######################################################################
# Crust

crust=readdlm("eos.table")
crust_n=crust[:,2]
crust_p=crust[:,4]
crust_e=(crust[:,5].+1).*939 .*crust_n;

function MatchCrusts()
    idx1=findmin(abs.(crust_n.-0.16))[2]
    idx2=findmin(abs.(crust_n.-0.16*0.7))[2]
    @show crust_n[idx1]/0.16
    @show crust_n[idx2]/0.16
    qmcidx1=findmin(abs.(crust_e[idx1]*mevpfm3 .-Etab))
    qmcidx2=findmin(abs.(crust_e[idx2]*mevpfm3 .-Etab))
    vEtab1=vcat(crust_e[1:idx1-1].*mevpfm3,Etab[qmcidx1[2]:end])
    vPtab1=vcat(crust_p[1:idx1-1].*mevpfm3,Ptab[qmcidx1[2]:end])

    vEtab2=vcat(crust_e[1:idx2-1].*mevpfm3,Etab[qmcidx2[2]:end])
    vPtab2=vcat(crust_p[1:idx2-1].*mevpfm3,Ptab[qmcidx2[2]:end])

    return vEtab1,vPtab1, vEtab2,vPtab2
end

(Etab1,Ptab1,Etab2,Ptab2)=MatchCrusts()
plot(crust_e,crust_p,marker=true,label="crust")
plot!(Etab./mevpfm3,Ptab./mevpfm3,marker=true,label="qmc")
plot!(xaxis=[0,200],yaxis=[0,10])
plot!(Etab1./mevpfm3,Ptab1./mevpfm3)
plot!(Etab2./mevpfm3,Ptab2./mevpfm3)

######################################################################
# TOV

function FD_nextstep(x,Fx,Gx,DFx,DGx,Δ)
    Fx_next= DFx(x,Fx,Gx)*Δ+Fx
    Gx_next= DGx(x,Fx,Gx)*Δ+Gx
    return Fx_next,Gx_next
end

const kmm2=1.3234e6*mevpfm3
const km=1.0
const fmm4_new=2.6115e-4/(km^2)
const G=1.0
const c=1.0
const Msol=1.4766km

Peos1=LinearInterpolation(Etab1[1:end].*fmm4_new,Ptab1[1:end].*fmm4_new,extrapolation_bc=Interpolations.Flat())
Eeos1=LinearInterpolation(Ptab1[1:end].*fmm4_new,Etab1[1:end].*fmm4_new,extrapolation_bc=Interpolations.Flat())
Peos2=LinearInterpolation(Etab2[1:end].*fmm4_new,Ptab2[1:end].*fmm4_new,extrapolation_bc=Interpolations.Flat())
Eeos2=LinearInterpolation(Ptab2[1:end].*fmm4_new,Etab2[1:end].*fmm4_new,extrapolation_bc=Interpolations.Flat())

∂P(r,Pr,Mr,ϵ)=-G*(ϵ(Pr)+Pr/c^2)*(Mr+4*π*Pr*(r)^3/c^2)/((r)^2*(1-2*G*Mr/((r)*c^2)))
∂M(r,Pr,Mr,ϵ)=4*π*(r)^2*ϵ(Pr)

function oneTOV(ϵ,P0,Rmax)
    δ=1e-3km
    p=P0
    r=0.0
    m=0.0
    resR=[r]
    resP=[p]
    resM=[m]
    while r<Rmax
        r=r+δ
        p=∂P(r,p,m,ϵ)*δ + p
        m=∂M(r,p,m,ϵ)*δ + m
        if p<=0.0
            push!(resP,p)
            push!(resM,m)
            push!(resR,r)
            break
        end

    end
    return resP,resM./Msol,resR,r
end

function MRall(ϵ,P0)
    M=Float64[]
    R=Float64[]
    for k in -2:0.02:1.3
        p1tov,m1tov,x,r1tov=oneTOV(ϵ,P0*10^k,100km)
        push!(M,m1tov[end])
        push!(R,r1tov)
    end
    @show maximum(M)
    return M,R
end

mtest1,rtest1=MRall(Eeos1,Ptab2[100].*fmm4_new)
mtest2,rtest2=MRall(Eeos2,Ptab2[100].*fmm4_new)

plot([0,23], [1.908,1.908],color=:gray,lw=3,label="PSR-J1714")
    plot!(rtest1,mtest1,color=:blue,label="QMC")
    plot!(rtest2,mtest2,color=:blue,label="")
    for i in 1:length(rtest1)
        plot!([rtest1[i],rtest2[i]],[mtest1[i],mtest2[i]],label="",color=:blue,lw=0.4)
    end
    tovplot=plot!(xaxis=[10,15],frame=:box,xlabel="r(km)",ylabel="M/M⊙")
    plot!([12.71+1.14,12.71+1.14,12.71-1.19,12.71-1.19,12.71+1.14],[1.34+0.15,1.34-0.16,1.34-0.16,1.34+0.15,1.34+0.15],lw=0.7,color=:gray,label="")
    plot!([13.02+1.24,13.02+1.24,13.02-1.06,13.02-1.06,13.02+1.24],[1.44+0.15,1.44-0.14,1.44-0.14,1.44+0.15,1.44+0.15],lw=0.7,color=:gray,label="")
    plot!([11.9+1.4,11.9+1.4,11.9-1.4,11.9-1.4,11.9+1.4],[1.36,1.6,1.6,1.36,1.36],lw=1.5,color=:black,label="")
    annotate!([11.1],[(1.6+1.36)/2+0.05],text("GW170817",9))
    annotate!([11.4],[(1.6+1.36)/2-0.05],text("90% confidence",9))
    annotate!([13.8],[1.53],text("NICER 2",9,:gray))
    annotate!([12.],[1.22],text("NICER 1",9,:gray))
savefig("tovplot.pdf")
