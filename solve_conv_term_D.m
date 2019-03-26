function d_hat = solve_conv_term_D(zhat_mat, zhat_inv_mat, d, B, rho, size_z )
%z��z���棨2 100 12100��ʽ��������Ҷ���µ�D��Bh{nn}�ֿ���ͼ��rho=500��z�Ĵ�С110 110 100 2��ÿһ�鶼����100��filters
    % Solves sum_j gamma_i/2 * || H_j d - xi_j ||_2^2
    % In our case: 1/2|| Zd - xi_1 ||_2^2 + rho * 1/2 * || d - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    n = size_z(end);%2
    ni=size_z(end);%2
    k = size_z(end - 1);%100
    ndim = length( size_z ) - 2;%2
    ss = prod(size_z(1:ndim));%���������ĳ˻�����110x110=12100
    
    xi_hat_1_cell = num2cell( permute( reshape(B, ss, ni), [2,1] ), 1);%��ͼ��ת��Ϊ2 12100�ĸ�ʽ��
    xi_hat_2_cell = num2cell( permute( reshape(d, ss, k), [2,1] ), 1);%������Ҷ���µ�d����100 12100����ʽ������֯
    
    %Invert
    x = cellfun(@(Sinv, A, b, c)(Sinv * (A' * b + rho * c)), zhat_inv_mat, zhat_mat,...
                                    xi_hat_1_cell, xi_hat_2_cell, 'UniformOutput', false);
                                %�������������dik+1
    
    %Reshape to get back the new Dhat
    ss_size = size_z(1:ndim);%��
    d_hat = reshape( permute(cell2mat(x), [2,1]), [ss_size,k] );%���12100 2����ʽȻ������µ�d
return;