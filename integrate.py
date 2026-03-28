#!/usr/bin/env python3
"""Numerical integration: trapezoidal, Simpson, Gauss-Legendre."""
import math
def trapezoidal(f,a,b,n=1000):
    h=(b-a)/n;s=f(a)+f(b)
    for i in range(1,n): s+=2*f(a+i*h)
    return s*h/2
def simpson(f,a,b,n=1000):
    if n%2: n+=1
    h=(b-a)/n;s=f(a)+f(b)
    for i in range(1,n):
        s+=(4 if i%2 else 2)*f(a+i*h)
    return s*h/3
def gauss_legendre(f,a,b,n=5):
    nodes_weights={2:[(-.5773502692,.5773502692),(1,1)],3:[(-.7745966692,0,.7745966692),(.5555555556,.8888888889,.5555555556)],5:[(-.9061798459,-.5384693101,0,.5384693101,.9061798459),(.2369268851,.4786286705,.5688888889,.4786286705,.2369268851)]}
    if n not in nodes_weights: n=5
    nodes,weights=nodes_weights[n]
    mid=(a+b)/2;half=(b-a)/2
    return half*sum(w*f(mid+half*x) for x,w in zip(nodes,weights))
def romberg(f,a,b,tol=1e-10,max_iter=20):
    R=[[0]*(max_iter+1) for _ in range(max_iter+1)]
    R[0][0]=trapezoidal(f,a,b,1)
    for i in range(1,max_iter+1):
        R[i][0]=trapezoidal(f,a,b,2**i)
        for j in range(1,i+1):
            R[i][j]=(4**j*R[i][j-1]-R[i-1][j-1])/(4**j-1)
        if i>1 and abs(R[i][i]-R[i-1][i-1])<tol: return R[i][i]
    return R[max_iter][max_iter]
if __name__=="__main__":
    f=math.sin;exact=1-math.cos(1)
    print(f"Trapezoidal: {trapezoidal(f,0,1):.10f}")
    print(f"Simpson: {simpson(f,0,1):.10f}")
    print(f"Gauss-5: {gauss_legendre(f,0,1):.10f}")
    print(f"Romberg: {romberg(f,0,1):.10f}")
    print(f"Exact: {exact:.10f}")
    assert abs(simpson(f,0,1)-exact)<1e-8
    print("Integration OK")
