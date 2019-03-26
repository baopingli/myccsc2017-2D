function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i
%参数(fft2(D{1})傅里叶域下的D，size_x 110 110 6
%Params
ndim = length( size_x ) - 1;%
ss = prod(size_x(1:ndim));%12100

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, ss, [] );%把傅里叶域下的d重构成12100的形式
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);%D的共轭乘上D 就是求zk+1时，左边的那一部分 就是DT*D

return;