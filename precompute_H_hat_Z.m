function [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(dhat, size_x )
% Computes the spectra for the inversion of all H_i
%����(fft2(D{1})����Ҷ���µ�D��size_x 110 110 6
%Params
ndim = length( size_x ) - 1;%
ss = prod(size_x(1:ndim));%12100

%Precompute the dot products for each frequency
dhat_flat = reshape( dhat, ss, [] );%�Ѹ���Ҷ���µ�d�ع���12100����ʽ
dhatTdhat_flat = sum(conj(dhat_flat).*dhat_flat,2);%D�Ĺ������D ������zk+1ʱ����ߵ���һ���� ����DT*D

return;