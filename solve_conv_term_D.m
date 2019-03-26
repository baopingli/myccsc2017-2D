function d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, d, B, rho, size_z )
%z、z的逆（2 100 12100格式）、傅里叶域下的D、Bh{nn}分块后的图像、rho=500、z的大小110 110 100 2，每一块都会有100个filters
    % Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
    % In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    n = size_z(end);%2
    ni=size_z(end);%2
    k = size_z(end - 1);%100
    ndim = length( size_z ) - 2;%2
    ss = prod(size_z(1:ndim));%返回向量的乘积就是110x110=12100
    
    xi_hat_1_cell = num2cell( permute( reshape(B, ss, ni), [2,1] ), 1);%将图像转换为2 12100的格式，
    xi_hat_2_cell = num2cell( permute( reshape(d, ss, k), [2,1] ), 1);%将傅里叶域下的d按照100 12100的形式重新组织
    
    %Invert
    x = cellfun(@(Sinv, A, b, c)(Sinv * (A' * b + rho * c)), zhat_inv_mat, zhat_mat,...
                                    xi_hat_1_cell, xi_hat_2_cell, 'UniformOutput', false);
                                %整个算出来就是dik+1
    
    %Reshape to get back the new Dhat
    ss_size = size_z(1:ndim);%将
    d_hat = reshape( permute(cell2mat(x), [2,1]), [ss_size,k] );%编程12100 2的形式然后就是新的d
return;