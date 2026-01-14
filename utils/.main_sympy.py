
import scipy as sp
import sympy as smp
from scipy.misc import derivative


zz_1, zz_2, a, b, c = smp.symbols('zz_1 zz_2 a b c', real=True)
f = smp.cos(a*zz_1)+smp.sin(b*zz_2)
print(f)

dfdx = smp.diff(f,zz_1)
print(dfdx)
dfdx = smp.diff(f,zz_2)
print(dfdx)