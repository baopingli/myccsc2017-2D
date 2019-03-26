function [u_proj] = KernelConstraintProj( u, size_d, psf_radius)

    %Params 110 110 100 5
    k = size_d(end);%100ge
    ndim = length( size_d ) - 1;%2

    %Get support
    u_proj = circshift( u, [psf_radius, psf_radius, 0] ); %将d进行循环移位 向右向下移位，就是将右下角的5x5的移到了左上角
    u_proj = u_proj(1:psf_radius*2+1,1:psf_radius*2+1,:);%取左上角的11x11。
    
     %Normalize 对d约束的 引入的指标函数的近端算子
 	u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), 1] );%将u_proj列相加然后，再行相加按照行列的大小进行平铺
    u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));%将里面所有的大于1的进行平均，能量的平均inc的近端算子
    
    %Now shift back and pad again
    u_proj = padarray( u_proj, [size_d(1:end - 1) - (2*psf_radius+1), 0], 0, 'post');%然后把其他的位置整成0
    u_proj = circshift(u_proj, -[repmat(psf_radius, 1, ndim), 0]);%然后按照原来的方式移位回去
    
return;