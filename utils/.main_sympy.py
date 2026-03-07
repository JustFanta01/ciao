
import sympy as sp
from sympy import init_printing
init_printing()


# parameters
alpha, beta, gamma, eta = sp.symbols(
    'alpha beta gamma eta', real=True
)

# constants (tutte positive se vuoi)
mu, rho = sp.symbols('mu rho', positive=True)
L1, L2, L3 = sp.symbols('L1 L2 L3', positive=True)
B, Bt = sp.symbols('B Bt', positive=True)   # norms
a = sp.symbols('a', positive=True)          # a = L2*||A-I||
c = sp.symbols('c', positive=True)          # c = ||(I-J)(I-L^2)|| < 1

M = sp.Matrix([

    [1 - mu*alpha,
     alpha*L1,
     alpha*L3,
     alpha*Bt,
     0],

    [alpha*L1*L3*(1+L3),
     rho + alpha*L1*L3,
     alpha*L3**2,
     alpha*L3*Bt,
     0],

    [alpha*L1*L2*(1+L3)**2,
     alpha*L1*L2*(1+L3) + a,
     rho + alpha*L2*L3*(1+L3),
     alpha*L2*(1+L3)*Bt,
     0],

    [beta*B*(1 - mu*alpha),
     alpha*beta*B*L1,
     alpha*beta*B*L3,
     c + alpha*beta*B*Bt,
     eta],

    [beta*gamma*B,
     alpha*beta*gamma*B*L1,
     alpha*beta*gamma*B*L3,
     gamma*(c + alpha*beta*B*Bt),
     1 - gamma*eta]
])

sp.pprint(M)

# detM = sp.factor(M.det())
# detM
