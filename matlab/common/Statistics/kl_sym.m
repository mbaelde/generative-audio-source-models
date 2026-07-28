function dist = kl_sym(p,q)
rp = repmat(p,[size(q,1),1]);

dist = sum(rp.*log(rp./q) + q.*log(q./rp),2);