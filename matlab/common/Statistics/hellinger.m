function dist = hellinger(p,q)

M = size(q,1);

dist = 0.5*sum((repmat(p,[M,1]) - q).^2,2);