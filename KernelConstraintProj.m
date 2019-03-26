function [u_proj] = KernelConstraintProj( u, size_d, psf_radius)

    %Params 110 110 100 5
    k = size_d(end);%100ge
    ndim = length( size_d ) - 1;%2

    %Get support
    u_proj = circshift( u, [psf_radius, psf_radius, 0] ); %��d����ѭ����λ ����������λ�����ǽ����½ǵ�5x5���Ƶ������Ͻ�
    u_proj = u_proj(1:psf_radius*2+1,1:psf_radius*2+1,:);%ȡ���Ͻǵ�11x11��
    
     %Normalize ��dԼ���� �����ָ�꺯���Ľ�������
 	u_norm = repmat( sum(sum(u_proj.^2, 1),2), [size(u_proj,1), size(u_proj,2), 1] );%��u_proj�����Ȼ��������Ӱ������еĴ�С����ƽ��
    u_proj( u_norm >= 1 ) = u_proj( u_norm >= 1 ) ./ sqrt(u_norm( u_norm >= 1 ));%���������еĴ���1�Ľ���ƽ����������ƽ��inc�Ľ�������
    
    %Now shift back and pad again
    u_proj = padarray( u_proj, [size_d(1:end - 1) - (2*psf_radius+1), 0], 0, 'post');%Ȼ���������λ������0
    u_proj = circshift(u_proj, -[repmat(psf_radius, 1, ndim), 0]);%Ȼ����ԭ���ķ�ʽ��λ��ȥ
    
return;