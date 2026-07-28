disp('Compiling...')
mex plcsSeparation.c functionsPLCS.c

disp('Initialize variables...')
F = 513;
n_channel = 2;
n_source = 5;
n_component = 200;

V = rand(F,n_channel);
P_f = rand(n_component,F);
iter_max = 50;

P_s = rand(n_source, n_channel);
P_t = rand(n_component, 1);

disp('Test routine...')
tic
mask = plcsSeparation(V,P_f,iter_max, P_s, P_t);
toc