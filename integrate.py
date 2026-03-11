#!/usr/bin/env python3
"""Numerical Integration — trapezoid, Simpson, Gauss-Legendre, Romberg."""
import math, sys

def trapezoid(f, a, b, n=1000):
    h = (b - a) / n
    return h * (f(a)/2 + sum(f(a + i*h) for i in range(1, n)) + f(b)/2)

def simpson(f, a, b, n=1000):
    if n % 2: n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    s += 4 * sum(f(a + i*h) for i in range(1, n, 2))
    s += 2 * sum(f(a + i*h) for i in range(2, n, 2))
    return s * h / 3

def gauss_legendre(f, a, b, n=5):
    # 5-point Gauss-Legendre
    nodes = [-0.9061798459, -0.5384693101, 0.0, 0.5384693101, 0.9061798459]
    weights = [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]
    mid = (a + b) / 2; half = (b - a) / 2
    return half * sum(w * f(mid + half * x) for x, w in zip(nodes[:n], weights[:n]))

def romberg(f, a, b, max_iter=10, tol=1e-12):
    R = [[0]*(max_iter+1) for _ in range(max_iter+1)]
    R[0][0] = (b - a) * (f(a) + f(b)) / 2
    for i in range(1, max_iter+1):
        h = (b - a) / (2**i)
        R[i][0] = R[i-1][0]/2 + h * sum(f(a + (2*k-1)*h) for k in range(1, 2**(i-1)+1))
        for j in range(1, i+1):
            R[i][j] = R[i][j-1] + (R[i][j-1] - R[i-1][j-1]) / (4**j - 1)
        if i > 1 and abs(R[i][i] - R[i-1][i-1]) < tol: return R[i][i]
    return R[max_iter][max_iter]

if __name__ == "__main__":
    f = math.sin; a, b = 0, math.pi; exact = 2.0
    print(f"∫sin(x)dx from 0 to π (exact: {exact})")
    print(f"  Trapezoid(100):  {trapezoid(f, a, b, 100):.12f}")
    print(f"  Simpson(100):    {simpson(f, a, b, 100):.12f}")
    print(f"  Gauss-Legendre:  {gauss_legendre(f, a, b, 5):.12f}")
    print(f"  Romberg:         {romberg(f, a, b):.12f}")
    # Harder: ∫e^(-x²)dx from 0 to ∞ ≈ √π/2
    g = lambda x: math.exp(-x**2)
    print(f"\n∫e^(-x²)dx [0,10] (exact≈{math.sqrt(math.pi)/2:.10f})")
    print(f"  Romberg: {romberg(g, 0, 10):.10f}")
