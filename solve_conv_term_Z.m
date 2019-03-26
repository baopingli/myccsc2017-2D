function z_hat = solve_conv_term_Z(dhatT, dhatTdhat, z, B, gammas, size_z )

%参数 DT  DT*D 频域z，傅里叶域的图像，50，110 110 100 6
    % Solves sum_j gamma_i/2 * || H_j z - xi_j ||_2^2
    % In our case: 1/2|| Dz - xi_1 ||_2^2 + rho * 1/2 * || z - xi_2||
    % with rho = gamma(2)/gamma(1)
    
    %Size
    ni = size_z(end);%6
    k = size_z(end - 1);%100
    ndim = length( size_z ) - 2;%2
    ss = prod(size_z(1:ndim));%12100
    
    %Rho
    rho = gammas;%50
    
    %Compute b
    b = dhatT .* permute( repmat( reshape(B, ss, 1, ni), [1,k,1] ), [2,1,3] ) + rho .* permute( reshape(z, ss, k, ni), [2,1,3] );
    %DT*b+rho*（y-lambda）
    %Invert
    z_hat = 1/rho *b - 1/rho * repmat( ones([1,ss]) ./ ( rho * ones([1,ss]) + dhatTdhat.' ), [k,1,ni] ) .* dhatT .* repmat( sum(conj(dhatT).*b, 1), [k,1,1] );
    %1/rho(DT*b+rho*（y-lambda）)-1/rho(I/(rho+DTD)*DT*D*D*b)没问题有b
    %Final transpose gives z_hat
    z_hat = reshape(permute(z_hat, [2,1,3]), size_z);

return;