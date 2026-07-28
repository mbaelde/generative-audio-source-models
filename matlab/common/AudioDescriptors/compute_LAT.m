function LogAttackTime = compute_LAT(AttackTime)
%% compute_LAT
%
% This function returns the log-attack time of a signal.
%
% LogAttackTime = compute_LAT(AttackTime)
%
% Author: Maxime BAELDE
% Date: 02/2016
% Company: A-Volute / INRIA

LogAttackTime = log10(diff(AttackTime));