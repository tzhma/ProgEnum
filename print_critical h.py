# fixed task parameters
dp, dpm = .3, -.5

# critical h
def print_h():
	h_c0 = 1/2 + (1-dpm)/(1+dp**2-dpm**2) - 4/(dp**2+(3-dpm)*(1+dpm))
	h_c1 = (1/(12*(dpm - 1)**3))*(2*(dpm - 1)*(dp**2 + 3*(dpm - 1)**2) + 2**(1/3)*(dp**6*(9*dpm - 5)*(dpm - 1)**3 - 18*dp**4*(dpm - 1)**6 + 9*dp**2*(dpm - 5)*(dpm - 1)**7 + 3*3**.5*((dpm - 1)**8*(-(dp**2 - (dpm - 1)**2)**2)*(dp**8*(2*dpm - 1) - 2*dp**6*(dpm - 1)**2*(4*dpm - 3) + dp**4*(dpm - 1)**2*(12*dpm**3 - 23*dpm**2 + 50*dpm - 23) - 8*dp**2*(dpm - 2)*(dpm - 1)**4*(dpm + 1)**2 + 2*(dpm - 1)**6*(dpm + 1)**3))**.5)**(1/3) + (2**(2/3)*(dp**4*(3*dpm - 1)*(dpm - 1)**2 - 6*dp**2*(dpm - 1)**5 + 3*(dpm + 1)*(dpm - 1)**6))/(dp**6*(9*dpm - 5)*(dpm - 1)**3 - 18*dp**4*(dpm - 1)**6 + 9*dp**2*(dpm - 5)*(dpm - 1)**7 + 3*3**.5*((dpm - 1)**8*(-(dp**2 - (dpm - 1)**2)**2)*(dp**8*(2*dpm - 1) - 2*dp**6*(dpm - 1)**2*(4*dpm - 3) + dp**4*(dpm - 1)**2*(12*dpm**3 - 23*dpm**2 + 50*dpm - 23) - 8*dp**2*(dpm - 2)*(dpm - 1)**4*(dpm + 1)**2 + 2*(dpm - 1)**6*(dpm + 1)**3))**.5)**(1/3))
	h_c2 = -dpm/(1-dpm)
	h_c3 = (1-dpm)/4 - dp**2/4/(1-dpm)
	print('h_c0=%s, h_c1=%s, h_c2=%s, h_c3=%s'%(h_c0, h_c1, h_c2, h_c3))

# special u
def print_u(h):
	u_cr = dp/(1-dpm)
	u_gw = (1-2*h) * dp/(1+dpm)
	u_gl = (1-2*h) * dp/(1-dpm)
	u_ub = ((1-2*h) + ((1+dpm)/dp*h)**2)**.5 - (1+dpm)/dp*h
	u_ll = (1-2*h) * (-dp+(1-dpm)*u_ub) / (-dp*u_ub+(1-dpm))
	print('u_cr=%s, u_gw=%s, u_gl=%s, u_ub=%s, u_ll=%s'%(u_cr, u_gw, u_gl, u_ub, u_ll))

# print
print_h()
print_u(h=.05)
