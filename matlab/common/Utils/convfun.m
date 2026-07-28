function f = convfun(t,x,phi,u,v)
f = betapdf(t/phi,u(1),v(1)).*betapdf((x-t)/(1-phi),u(2),v(2));