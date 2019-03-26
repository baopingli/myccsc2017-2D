function [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, size_z, rho)
% Computes the spectra for the inversion of all H_i

%Params  size_z 110 110 11 100 2
n = size_z(end);%2
ni = size_z(end);%2
k = size_z(end - 1);%100
ndim = length( size_z ) - 2;%3
ss = prod(size_z(1:ndim));%����110x110x11,����һ�еĳ˻�����������в���1��ʱ�򣬼����еĳ˻�

%Precompute spectra for H 
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [ss, k, ni] ), [3,2,1] ), [1 2] ), [1 ss]); %n * k * s
%ss k ni:12100 100 2 ���ǰ�z�ĸ���Ҷ���µ�һ�У�����һ��Z��Ȼ��ά���û��ɳ�2 100 12100��Ȼ��ת��Ϊcell����ϡ�
%Precompute the inverse matrices for each frequency
%Ȼ�����Z�ľ����棨ZtZ+pI��-1������Ǹ�ʽ��
zhat_inv_mat = reshape( cellfun(@(A)(1/rho * eye(k) - 1/rho * A'*pinv(rho * eye(ni) + A * A')*A), zhat_mat, 'UniformOutput', false'), [1 ss]);

return;