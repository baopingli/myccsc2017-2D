function [zhat_mat, zhat_inv_mat] = precompute_H_hat_D(z_hat, size_z, rho)
% Computes the spectra for the inversion of all H_i

%Params  size_z 110 110 11 100 2
n = size_z(end);%2
ni = size_z(end);%2
k = size_z(end - 1);%100
ndim = length( size_z ) - 2;%3
ss = prod(size_z(1:ndim));%计算110x110x11,计算一列的乘积、如果后面有参数1的时候，计算行的乘积

%Precompute spectra for H 
zhat_mat = reshape( num2cell( permute( reshape(z_hat, [ss, k, ni] ), [3,2,1] ), [1 2] ), [1 ss]); %n * k * s
%ss k ni:12100 100 2 就是把z的傅里叶域下的一行，就是一个Z，然后维度置换成成2 100 12100，然后转换为cell的组合。
%Precompute the inverse matrices for each frequency
%然后计算Z的矩阵逆（ZtZ+pI）-1是最后那个式子
zhat_inv_mat = reshape( cellfun(@(A)(1/rho * eye(k) - 1/rho * A'*pinv(rho * eye(ni) + A * A')*A), zhat_mat, 'UniformOutput', false'), [1 ss]);

return;