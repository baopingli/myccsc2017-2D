actual_files = 0;
image=1;
image_path='./Data/';%��ǰĿ¼�µ�data�ļ���
image_frames={1,1,'end'};
if(ischar(image_frames{3}))
    last_ind=sprintf('%d:%d:%s',image_frames{1},image_frames{2},image_frames{3});
else
    last_ind=sprintf('%d:%d:%d',image_frames{1},image_frames{2},image_frames{3});
end
fprintf('going to select frames:%s',last_ind);
subdir=dir(image_path)
[~,files]=split_folders_files(subdir); %�õ���file���������һ��
fprintf('The length of the I file cell array found in this directory is: %d\n',length(files));
%�м���Ҫ�ж�һ������end��Ӧ�ķ�Χ�ǲ��ǳ������ļ��е�ͼƬ�ĸ���������ʡ�ԡ�
%Ȼ��������ͼƬ����ѭ��
if(ischar(image_frames{3}))
        image_frames{3} = length(files);
else
        if(image_frames{3}>length(files))
            image_frames{3}=length(files);
        end
end

for file=image_frames{1}:image_frames{2}:image_frames{3}
    if(files(file).isdir==0)
        %��ȡ�ļ�·��
        img_name=strcat(image_path,files(file).name);
        try
            %loadͼ��
            IMG=single(imread(img_name)); 
            actual_files=actual_files+1;
            fprintf('Loading: %s \n Image: %10d/%10d. Selecting every %5d. Selected: %10d so far.\r',img_name,file,length(files),image_frames{2},actual_files)
            if(actual_files==1)
                    I = cell(1,length(files));%�½�cell
            end
            I{actual_files}=IMG;
            image=image+1;
        catch
            fprintf('Counld not load %s as an image.\n',img_name);
        end
    end
end
I=I(1:actual_files);%����I���ǰ���6��ͼƬ��cell������
%Ȼ����лҶȴ���
max(I{1}(:))  %���ֵ��255
min(I{1}(:))  %��Сֵ��9
for i=1:length(I)
    fprintf('making gray image %10d\r',i); 
    if(size(I{i},3)==3)%˵���ǲ�ɫͼƬ
        I{i}=single(rgb2gray(double(I{i})/255.0));%˫�����³���255.0Ȼ����ת��Ϊ������,
%         subplot(1,1,i)
%         imshow(I{i},[9,200])
        figure(1)
        subplot(1,6,i);
        imagesc(im2double(I{i}));
        axis image
    else
        if(max(I{i}(:))>1)
            I{i}=double(I{i})/255.0;
        else
            I{i}=double(I{i});
        end
    end
end
%Ȼ����оֲ��Աȶȹ�һ��
num_colors=size(I{1},3);%1
k=fspecial('gaussian',[13 13],3*1.591);%������˹��ͨ�˲������ߴ�Ϊ13x13��3*1.591�Ǳ�׼��
k2=fspecial('gaussian',[13 13],3*1.591);
 if(all(k(:)==k2(:)))
            SAME_KERNELS=1;
        else
            SAME_KERNELS=0;
 end
 %�Աȶȹ�һ���Ĺ���
 for image=1:length(I)
            fprintf('Contrast Normalizing Image with Local CN: %10d\r',image);
            temp = I{image};
            for j=1:num_colors
                %                 if(image==151)
                %                     keyboard
                %                 end
                dim = double(temp(:,:,j));%ֻ��һά�����ڽ�����double
                %                 lmn = conv2(dim,k,'valid');
                %                 lmnsq = conv2(dim.^2,k,'valid');
                lmn = rconv2(dim,k);%���
                figure(2);
                subplot(3,6,image);
                imagesc(lmn);
                axis image
                lmnsq = rconv2(dim.^2,k2);%ÿ�����ؽ���ƽ��Ȼ����
                subplot(3,6,image+6);
                imagesc(lmnsq);
                axis image
                if(SAME_KERNELS)
                    lmn2 = lmn;
                else
                    lmn2 = rconv2(dim,k2);
                end
                lvar = lmnsq - lmn2.^2;%�;��֮���ƽ������õ�lvar
                
                subplot(3,6,image+12);
                imagesc(lvar);
                axis image
                
                lvar(lvar<0) = 0; % avoid numerical problems,��С��0����Ϊ0
                lstd = sqrt(lvar);%lstd��sqrt�Ŀ���
                
                q=sort(lstd(:));%Ĭ�Ͻ�����������
                lq = round(length(q)/2);%������������м����ص�λ��
                th = q(lq);%�м����ض�Ӧ��ֵ
                if(th==0)
                    q = nonzeros(q);%�ҳ������з����Ԫ��
                    if(~isempty(q))%����ַ����Ԫ��
                    lq = round(length(q)/2);%������ֵ
                    th = q(lq);%������Ӧ���м����ص�ֵӦ�ò���0��
                    else
                        th = 0;
                    end
                end
                lstd(lstd<=th) = th;%�ò�ֵ������ľ����е�����ֵС��th��ȫ������th
                %lstd(lstd<(8/255)) = 8/255;
                %                 lstd = conv2(lstd,k2,'same');
                
                
                lstd(lstd(:)==0) = eps;
                
                %                 shifti = floor(size(k,1)/2)+1;
                %                 shiftj = floor(size(k,2)/2)+1;
                
                % since we do valid convolutions
                %                 dim = dim(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1);
                dim = dim - lmn;%�Ҷ�ͼ���ȥ��˹��ͨ�˲����ͼ��
                dim = dim ./ lstd;%Ȼ��ÿ������ֵ����������ͨ��ֵȻ�󿪷����ֵ��
                
                temp(:,:,j) = dim;
                %                 res_I{image}(:,:,j) = single(double(I{image}(:,:,j))-dim);
                %                 res_I{image}(:,:,j) = double(I{image}(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1,j))-double(CN_I{image}(:,:,j));  % Compute the residual image.
                %             IMG = conI;
            end
            I{image} = single(temp);
 end       
%Ȼ�����0��ֵ��
for i=1:length(I)
    fprintf('making image %10d Zero Mean.\r',i);
    I{i}=I{i}-mean(I{i}(:));
end

% Now all of I is assumed to be the same size.
[xdim,ydim,colors] = size(I{1});
numims = length(I);
% Make sure it is a row vector.
I = reshape(I,[1 numims]);

I = single(cell2mat(I));%��cell�еĶ������ϳ�һ��

figure(3);
imagesc(im2double(I));
axis image

I = reshape(I,[xdim ydim numims colors]);%100,100,6,1
I = permute(I,[1 2 4 3]);%����3 4ά
I = double(I);

%     I = reshape(I,[xdim ydim colors numims]);


fprintf('Not Squaring, just converting all images from cell to matrix...\n')
%     for image=1:length(I)
%         I(:,:,:,image) = single(I{image});
%     end
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nAll Images have been loaded and preprocessed.\n\n');
I = reshape(I, size(I,1), size(I,2), [] ) ;

whos I
%Ȼ��Ϳ�ʼ�˽���������ȡ��
%�����˲����Ĵ�С
verbose='all'
kernel_size=[11,11,100];
lambda_residual=1.0;
lambda_prior=1.0;%����lambda
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d] kernels.\n\n', kernel_size(3), kernel_size(1), kernel_size(2))
%�Ż�Ŀ��
verbose_admm = 'all';
max_it = 20;%����������
tol = 1e-3;%������0.001
tic();
%admm learn 2D������
psf_s=kernel_size(1);%11
k=kernel_size(end);%100��
sb=size(I);%sb��I�Ĵ�С100x100x1x6
n=sb(end);%6��ͼ��
ni=2;%ÿ�ε�������ͼƬ ni����˼�Ƿֳɶ��ٷݣ��ֳɵķ���Խ��ռ�õ��ڴ�ԽС��
N=n/ni;%3 ����������ε���

psf_radius=floor(psf_s/2);%5
size_x=[sb(1:end-1)+2*psf_radius,n];%110 110  6
size_z=[size_x(1:end-1),k,n];%110 110 100 6
size_z_crop=[size_x(1:end-1),k,ni];%110 110  100 2
size_d_full=[size_x(1:end-1),k];%110 110  100
lambda=[lambda_residual,lambda_prior];%1.0 1.0
B=padarray(I,[psf_radius,psf_radius,0],0,'both');%��I����110 110 ���ܱ�ȫ��0������
B_hat=fft2(B);%ͼ��ĸ���Ҷ�仯 110 110  6
size(B_hat)
%�ֿ����������ͼƬΪһ�飬�Ȱ����е�ͼ��ϳ�Ϊһ��Ȼ���ٽ��зֿ�Ĳ���
for nn=1:N
    Bh{nn}=B_hat(:,:,(nn-1)*ni+1:nn*ni);%110 110 2�Ĵ�С
    size(Bh{nn})
end
%��������
ProxSparse=@(u,theta) max(0,1-theta./abs(u)).*u;%1-thea*beita/|v|,ϡ��Ľ�������
ProxKernelConstraint=@(u)KernelConstraintProj(u,size_d_full,psf_radius);%110 110 100 //// 5
objective=@(z,d)objectiveFunction(z, d, I, lambda_residual, lambda_prior, psf_radius, size_z, size_x);
 d = padarray( randn(kernel_size), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2),0], 0, 'post');%ÿһά���һ��Ԫ�����
 size(d)%110 110 100һ����100��
 d = circshift(d, -[psf_radius, psf_radius, 0] );%����������λ11�У�������ܶ��о��������
 size(d)
 d_hat=fft2(d);
 dup=repmat({d_hat},N,1);%��110x110ƽ�̳�3��һ�У���Ϊ�ֳ���3������Ҫ�������ε�����ÿ������ͼƬ�����������d��������d��d����ƽ�̣�
 size(dup)
 D=repmat({d},N,1);%D��ʱ���µ�d
 z=randn(size_z);%100��110x110��z 110 110 100 6
 z_hat=fft2(z);%110 110 100 6
 %���������ʾ�Ĳ���
 if  strcmp( verbose, 'all')
        iterate_fig = figure();
        filter_fig = figure();
        display_func(iterate_fig, filter_fig, d, z_hat, I, size_x, size_z_crop, psf_radius, 0);
 end
 if strcmp( verbose, 'brief') || strcmp(verbose, 'all')
        obj_val = objective(z, d);
        fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)
        obj_val_filter = obj_val;
        obj_val_z = obj_val;
 end
 
 %�������е��Ż�ֵ�ͻ��ѵ�ʱ��
 iterations.obj_vals_d=[];
 iterations.obj_vals_z=[];
 iterations.tim_vals=[];
 %�������еĳ�ʼ����
 iterations.obj_vals_d(1)=obj_val_filter;
 iterations.obj_vals_z(1)=obj_val_z;
 iterations.tim_vals(1)=0;
 %������
 max_it_d=10;
 max_it_z=10;
 Dbar = zeros(size_d_full);%110 110 100��С��0����
 Udbar = zeros(size_d_full);%ͬ��Ҳ��110 110 100��С��0����
 d_D = repmat({zeros(size_d_full)},N,1);%������ƽ�� 3��
 d_Z =  zeros(size_z);
 %������ʼ
 for i=1:max_it
     tic;
     %Ŀ�����D
     for nn=1:N
         fprintf('Starting D preprocessing iterations: %d! \n', nn);
         zup{nn} = z_hat(:,:,:,(nn-1)*ni + 1:nn*ni) ;%12 34 56 ������Ӧ��Z�Ѿ�ȷ����ʱ��ȥ��������
         [zhat_mat{nn}, zhat_inv_mat{nn}] = precompute_H_hat_D(zup{nn}, size_z_crop, 500); %gammas_D(2)/gammas_D(1),�������������棬
     end
     t_kernel=toc;
     fprintf('staring D iterations after preprocessing!\n')
     %һ���ܵĵ�����20�Σ�ÿ�ε�����ʱ������d��z�ֱ����10��
     for i_d=1:max_it_d
         d_old=D{1};
         tic;
         u_D2=ProxKernelConstraint(Dbar+Udbar);%inc��һ�� yk+1 ����������һ��
         for nn=1:N
             d_D{nn}=d_D{nn}+(D{nn}-u_D2);%d_D���ǽ�������Լ��֮���D namd=namd+d-yk+1
             ud_D{nn}=fft2(u_D2-d_D{nn});%���������D�ĸ���Ҷ�仯 yk+1-namd   ud_D{n}����yk-namdk
             dup{nn}=solve_conv_term_D(zhat_mat{nn},zhat_inv_mat{nn},ud_D{nn},Bh{nn},500,size_z_crop);%�����ܵ�һ��ʽ��
             D{nn}=real(ifft2(dup{nn}));%���յ�D
         end
         Dbar =0; Udbar = 0;
            for nn=1:N
                Dbar = Dbar + D{nn};%�����е�filters����ÿһ����õ�d
                Udbar = Udbar + d_D{nn};%namd
            end
            Dbar = (1/N)*Dbar;%��d��ƽ��
            Udbar = (1/N)*Udbar;%��namd��ƽ��
            
            t_kernel_tmp = toc;
            t_kernel = t_kernel + t_kernel_tmp;%��ʱ��
            
            d_diff = D{1} - d_old;%������һ��d�ı仯
            if strcmp(verbose, 'brief')
                obj_val_filter = objective(z, D{1});%�����Ż������������ռ������Сֵ
                fprintf('Iter D %d, Obj %3.3g, Diff %5.5g\n', i_d, obj_val_filter, norm(d_diff(:),2)/ norm(D{1}(:),2));
                %����������һ���൱�� d�Ĳ�ֵ�Ķ�����/ԭd�Ķ����� �������ֵ
            end
            if (norm(d_diff(:),2)/ norm(D{1}(:),2) < tol)
                break;
            end
            %˵��d�Ѿ������ȶ������ԾͿ���break����������z�ĵ���
         
     end
      if  strcmp( verbose, 'all')
            display_func(iterate_fig, filter_fig, D{1}, z_hat, I, size_x, size_z_crop, psf_radius, i);
      end   
      
      %Z�ĵ���
      tic;
      fprintf('strating Z preprocessing iterations:!\n');
      [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(fft2(D{1}), size_x);%DTD
      dhatT_flat = repmat(  conj(dhat_flat.'), [1,1,n] );%ת��Ȼ�������Ĺ��DT
      t_vars=toc;%��һ�������Z����ߵ���Ĳ��ֵ�ʱ�����õ�ʱ��
      for i_z=1:max_it_z
          z_old=z;%z�����ܵ�Ŀ�꺯���е�z 110 110 100 6
          tic;
          %�����u_Z2����y����Ӧ����Ҫ��ʾ���ݶȵġ�
          %TV�����z����Ӧ��ת�����ݶȣ�Ȼ�����
          %��110x110x6x100��Ȼ��ֳ�����һ��x�����һ��y�����grad
          %Ȼ��ֱ�ȥ����Լ��������ֵ������
          %Ȼ���ٺ���һ����y���뵽z���ܵ�ʽ����Ȼ�����z
          u_Z2=ProxSparse(z+d_Z,lambda(2)/50);%�˴�lambda(2)��1.0�Ǽ��������е�thea Ϊ�˱�֤ϡ��Ľ������ӡ� �൱��y��
          d_Z=d_Z+(z-u_Z2);%d_Z�Ƕ�Ӧ��admm�е�lambda=lambda+z-yik+1���ȶ�z����Լ��Ȼ���ٴ�����Z�ĵ�������
          ud_Z = fft2( u_Z2 - d_Z ) ;%ud_Z��y-lambda
          z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, ud_Z, B_hat, 50, size_z);
          z=real(ifft2(z_hat));%z��Ϊʱ����
          
           t_vars_tmp = toc;
           t_vars = t_vars + t_vars_tmp;
           z_diff = z - z_old;
           if strcmp( verbose, 'brief')
                obj_val_z = objective(z, D{1});
                fprintf('Iter Z %d, Obj %3.3g, Diff %5.5g\n', i_z, obj_val_z, norm(z_diff(:),2)/ norm(z(:),2)) %  
            end
            if (norm(z_diff(:),2)/ norm(z(:),2) < tol)
                break;
            end
           
            
      end
      if strcmp(verbose,'all')
          display_func(iterate_fig, filter_fig, D{1}, z_hat, I, size_x, size_z_crop, psf_radius, i);
          fprintf('Sprase coding learing loop:%d\n\n',i);
      end
        iterations.obj_vals_d(i + 1) = obj_val_filter;
        iterations.obj_vals_z(i + 1) = obj_val_z;
        iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_kernel + t_vars;
       if norm(z_diff(:),2)/ norm(z(:),2) < tol && norm(d_diff(:),2)/ norm(D{1}(:),2) < tol
            break;
       end       
 end
    DZ = real(ifft2( sum(z_hat.* repmat(dup{1}, 1,1,1,n),3) ));    
    d_res = circshift(D{1}, [psf_radius, psf_radius, 0] );
    d_res = d_res(1:psf_radius*2+1,1:psf_radius*2+1, :);
    z_res = z;  
 
 tt=toc;
 %��ʾ���
 psf_radius = 5;
figure();    
pd = 1;
sqr_k = ceil(sqrt(size(d,3)));
d_disp = zeros( sqr_k * [kernel_size(1) + pd, kernel_size(2) + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (kernel_size(1) + pd) + pd + (1:kernel_size(1)) , mod(j,sqr_k) * (kernel_size(2) + pd) + pd + (1:kernel_size(2)) ) = d(:,:,j + 1); 
end
imagesc(d_disp), colormap gray, axis image, colorbar, title('Final filter estimate');
 prefix = 'ours';
save(sprintf('Filters_%s_2D_large.mat', prefix), 'd', 'Dz', 'iterations');
fprintf('Done sparse coding learning! --> Time %2.2f sec.\n\n', tt)

 













        
        
        

        

