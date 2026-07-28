function f = fun(t,phi,x,u,v)

f = t.^(u(1)-1) .* (x-t).^(u(2)-1) .* (phi-t).^(v(1)-1) .* (1-phi-x+t).^(v(2)-1);