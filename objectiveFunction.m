function f_val = objectiveFunction(z, d, b, lambda_residual, lambda, psf_radius, size_z, size_x)
    
    %Params
    n = size_x(end);
    k = size_z(end-1);
    ndim = length( size_z ) - 2;
    Dz = zeros( size_x );
    all_dims = repmat(':,',1,ndim);
    
    
    Dz = real(ifft2( sum(fft2(z).* repmat(fft2(d), 1,1,1,n),3) ));
    f_z = lambda_residual * 1/2 * norm( reshape(  Dz(1 + psf_radius:end - psf_radius,1 + psf_radius:end - psf_radius,:) - b, [], 1) , 2 )^2; 
            
    g_z = lambda * sum( abs( z(:) ), 1 );
    
    %Function val
    f_val = f_z + g_z;
    
    
return;