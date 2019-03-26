actual_files = 0;
image=1;
image_path='./Data/';%当前目录下的data文件夹
image_frames={1,1,'end'};
if(ischar(image_frames{3}))
    last_ind=sprintf('%d:%d:%s',image_frames{1},image_frames{2},image_frames{3});
else
    last_ind=sprintf('%d:%d:%d',image_frames{1},image_frames{2},image_frames{3});
end
fprintf('going to select frames:%s',last_ind);
subdir=dir(image_path)
[~,files]=split_folders_files(subdir); %得到了file可以输出看一下
fprintf('The length of the I file cell array found in this directory is: %d\n',length(files));
%中间需要判断一下输入end对应的范围是不是超过了文件中的图片的个数，这里省略。
%然后对里面的图片进行循环
if(ischar(image_frames{3}))
        image_frames{3} = length(files);
else
        if(image_frames{3}>length(files))
            image_frames{3}=length(files);
        end
end

for file=image_frames{1}:image_frames{2}:image_frames{3}
    if(files(file).isdir==0)
        %获取文件路径
        img_name=strcat(image_path,files(file).name);
        try
            %load图像
            IMG=single(imread(img_name)); 
            actual_files=actual_files+1;
            fprintf('Loading: %s \n Image: %10d/%10d. Selecting every %5d. Selected: %10d so far.\r',img_name,file,length(files),image_frames{2},actual_files)
            if(actual_files==1)
                    I = cell(1,length(files));%新建cell
            end
            I{actual_files}=IMG;
            image=image+1;
        catch
            fprintf('Counld not load %s as an image.\n',img_name);
        end
    end
end
I=I(1:actual_files);%现在I就是包含6张图片的cell数组了
%然后进行灰度处理
max(I{1}(:))  %最大值是255
min(I{1}(:))  %最小值是9
for i=1:length(I)
    fprintf('making gray image %10d\r',i); 
    if(size(I{i},3)==3)%说明是彩色图片
        I{i}=single(rgb2gray(double(I{i})/255.0));%双精度下除以255.0然后在转换为单精度,
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
%然后进行局部对比度归一化
num_colors=size(I{1},3);%1
k=fspecial('gaussian',[13 13],3*1.591);%创建高斯低通滤波器，尺寸为13x13，3*1.591是标准差
k2=fspecial('gaussian',[13 13],3*1.591);
 if(all(k(:)==k2(:)))
            SAME_KERNELS=1;
        else
            SAME_KERNELS=0;
 end
 %对比度归一化的过程
 for image=1:length(I)
            fprintf('Contrast Normalizing Image with Local CN: %10d\r',image);
            temp = I{image};
            for j=1:num_colors
                %                 if(image==151)
                %                     keyboard
                %                 end
                dim = double(temp(:,:,j));%只有一维，现在进行了double
                %                 lmn = conv2(dim,k,'valid');
                %                 lmnsq = conv2(dim.^2,k,'valid');
                lmn = rconv2(dim,k);%卷积
                figure(2);
                subplot(3,6,image);
                imagesc(lmn);
                axis image
                lmnsq = rconv2(dim.^2,k2);%每个像素进行平方然后卷积
                subplot(3,6,image+6);
                imagesc(lmnsq);
                axis image
                if(SAME_KERNELS)
                    lmn2 = lmn;
                else
                    lmn2 = rconv2(dim,k2);
                end
                lvar = lmnsq - lmn2.^2;%和卷积之后的平方做差得到lvar
                
                subplot(3,6,image+12);
                imagesc(lvar);
                axis image
                
                lvar(lvar<0) = 0; % avoid numerical problems,将小于0的置为0
                lstd = sqrt(lvar);%lstd是sqrt的开方
                
                q=sort(lstd(:));%默认进行升序排序
                lq = round(length(q)/2);%四舍五入求出中间像素的位置
                th = q(lq);%中间像素对应的值
                if(th==0)
                    q = nonzeros(q);%找出矩阵中非零的元素
                    if(~isempty(q))%如果又非零的元素
                    lq = round(length(q)/2);%再找中值
                    th = q(lq);%这样对应的中间像素的值应该不是0了
                    else
                        th = 0;
                    end
                end
                lstd(lstd<=th) = th;%让差值开方后的矩阵中的像素值小于th的全部等于th
                %lstd(lstd<(8/255)) = 8/255;
                %                 lstd = conv2(lstd,k2,'same');
                
                
                lstd(lstd(:)==0) = eps;
                
                %                 shifti = floor(size(k,1)/2)+1;
                %                 shiftj = floor(size(k,2)/2)+1;
                
                % since we do valid convolutions
                %                 dim = dim(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1);
                dim = dim - lmn;%灰度图像减去高斯低通滤波后的图像
                dim = dim ./ lstd;%然后每个像素值除以两个高通差值然后开方后的值。
                
                temp(:,:,j) = dim;
                %                 res_I{image}(:,:,j) = single(double(I{image}(:,:,j))-dim);
                %                 res_I{image}(:,:,j) = double(I{image}(shifti:shifti+size(lstd,1)-1,shiftj:shiftj+size(lstd,2)-1,j))-double(CN_I{image}(:,:,j));  % Compute the residual image.
                %             IMG = conI;
            end
            I{image} = single(temp);
 end       
%然后进行0均值化
for i=1:length(I)
    fprintf('making image %10d Zero Mean.\r',i);
    I{i}=I{i}-mean(I{i}(:));
end

% Now all of I is assumed to be the same size.
[xdim,ydim,colors] = size(I{1});
numims = length(I);
% Make sure it is a row vector.
I = reshape(I,[1 numims]);

I = single(cell2mat(I));%把cell中的多个矩阵合成一个

figure(3);
imagesc(im2double(I));
axis image

I = reshape(I,[xdim ydim numims colors]);%100,100,6,1
I = permute(I,[1 2 4 3]);%交换3 4维
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
%然后就开始了进行特征提取了
%定义滤波器的大小
verbose='all'
kernel_size=[11,11,100];
lambda_residual=1.0;
lambda_prior=1.0;%先验lambda
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d] kernels.\n\n', kernel_size(3), kernel_size(1), kernel_size(2))
%优化目标
verbose_admm = 'all';
max_it = 20;%最大迭代次数
tol = 1e-3;%定义了0.001
tic();
%admm learn 2D的内容
psf_s=kernel_size(1);%11
k=kernel_size(end);%100个
sb=size(I);%sb是I的大小100x100x1x6
n=sb(end);%6张图像
ni=2;%每次迭代两张图片 ni的意思是分成多少份，分成的份数越多占用的内存越小，
N=n/ni;%3 里面进行三次迭代

psf_radius=floor(psf_s/2);%5
size_x=[sb(1:end-1)+2*psf_radius,n];%110 110  6
size_z=[size_x(1:end-1),k,n];%110 110 100 6
size_z_crop=[size_x(1:end-1),k,ni];%110 110  100 2
size_d_full=[size_x(1:end-1),k];%110 110  100
lambda=[lambda_residual,lambda_prior];%1.0 1.0
B=padarray(I,[psf_radius,psf_radius,0],0,'both');%把I填充成110 110 的周边全是0的那种
B_hat=fft2(B);%图像的傅里叶变化 110 110  6
size(B_hat)
%分块操作，两张图片为一块，先把所有的图像合称为一块然后再进行分块的操作
for nn=1:N
    Bh{nn}=B_hat(:,:,(nn-1)*ni+1:nn*ni);%110 110 2的大小
    size(Bh{nn})
end
%近端算子
ProxSparse=@(u,theta) max(0,1-theta./abs(u)).*u;%1-thea*beita/|v|,稀疏的近端算子
ProxKernelConstraint=@(u)KernelConstraintProj(u,size_d_full,psf_radius);%110 110 100 //// 5
objective=@(z,d)objectiveFunction(z, d, I, lambda_residual, lambda_prior, psf_radius, size_z, size_x);
 d = padarray( randn(kernel_size), [size_x(1) - kernel_size(1), size_x(2) - kernel_size(2),0], 0, 'post');%每一维最后一个元素填充
 size(d)%110 110 100一共有100个
 d = circshift(d, -[psf_radius, psf_radius, 0] );%向上向左移位11列，变成四周都有矩阵的样子
 size(d)
 d_hat=fft2(d);
 dup=repmat({d_hat},N,1);%将110x110平铺成3行一列，因为分成了3个部分要进行三次迭代，每次两张图片，所以这个是d将卷积后的d和d进行平铺，
 size(dup)
 D=repmat({d},N,1);%D是时域下的d
 z=randn(size_z);%100个110x110的z 110 110 100 6
 z_hat=fft2(z);%110 110 100 6
 %交代最后显示的操作
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
 
 %保存所有的优化值和花费的时间
 iterations.obj_vals_d=[];
 iterations.obj_vals_z=[];
 iterations.tim_vals=[];
 %保存所有的初始变量
 iterations.obj_vals_d(1)=obj_val_filter;
 iterations.obj_vals_z(1)=obj_val_z;
 iterations.tim_vals(1)=0;
 %迭代大法
 max_it_d=10;
 max_it_z=10;
 Dbar = zeros(size_d_full);%110 110 100大小的0矩阵
 Udbar = zeros(size_d_full);%同样也是110 110 100大小的0矩阵
 d_D = repmat({zeros(size_d_full)},N,1);%进行了平铺 3个
 d_Z =  zeros(size_z);
 %迭代开始
 for i=1:max_it
     tic;
     %目测迭代D
     for nn=1:N
         fprintf('Starting D preprocessing iterations: %d! \n', nn);
         zup{nn} = z_hat(:,:,:,(nn-1)*ni + 1:nn*ni) ;%12 34 56 三个对应的Z已经确定的时候，去求矩阵的逆
         [zhat_mat{nn}, zhat_inv_mat{nn}] = precompute_H_hat_D(zup{nn}, size_z_crop, 500); %gammas_D(2)/gammas_D(1),求出三个矩阵的逆，
     end
     t_kernel=toc;
     fprintf('staring D iterations after preprocessing!\n')
     %一共总的迭代是20次，每次迭代的时候里面d和z分别迭代10次
     for i_d=1:max_it_d
         d_old=D{1};
         tic;
         u_D2=ProxKernelConstraint(Dbar+Udbar);%inc那一块 yk+1 近端算子那一块
         for nn=1:N
             d_D{nn}=d_D{nn}+(D{nn}-u_D2);%d_D就是进行能量约束之后的D namd=namd+d-yk+1
             ud_D{nn}=fft2(u_D2-d_D{nn});%就是上面的D的傅里叶变化 yk+1-namd   ud_D{n}就是yk-namdk
             dup{nn}=solve_conv_term_D(zhat_mat{nn},zhat_inv_mat{nn},ud_D{nn},Bh{nn},500,size_z_crop);%计算总的一个式子
             D{nn}=real(ifft2(dup{nn}));%最终的D
         end
         Dbar =0; Udbar = 0;
            for nn=1:N
                Dbar = Dbar + D{nn};%把所有的filters加上每一次算好的d
                Udbar = Udbar + d_D{nn};%namd
            end
            Dbar = (1/N)*Dbar;%对d求平均
            Udbar = (1/N)*Udbar;%对namd求平均
            
            t_kernel_tmp = toc;
            t_kernel = t_kernel + t_kernel_tmp;%算时间
            
            d_diff = D{1} - d_old;%计算了一下d的变化
            if strcmp(verbose, 'brief')
                obj_val_filter = objective(z, D{1});%最后的优化函数就是最终计算的最小值
                fprintf('Iter D %d, Obj %3.3g, Diff %5.5g\n', i_d, obj_val_filter, norm(d_diff(:),2)/ norm(D{1}(:),2));
                %最后输出的这一项相当于 d的差值的二范数/原d的二范数 算出来的值
            end
            if (norm(d_diff(:),2)/ norm(D{1}(:),2) < tol)
                break;
            end
            %说明d已经趋于稳定了所以就可以break迭代。进行z的迭代
         
     end
      if  strcmp( verbose, 'all')
            display_func(iterate_fig, filter_fig, D{1}, z_hat, I, size_x, size_z_crop, psf_radius, i);
      end   
      
      %Z的迭代
      tic;
      fprintf('strating Z preprocessing iterations:!\n');
      [dhat_flat, dhatTdhat_flat] = precompute_H_hat_Z(fft2(D{1}), size_x);%DTD
      dhatT_flat = repmat(  conj(dhat_flat.'), [1,1,n] );%转置然后求复数的共轭，DT
      t_vars=toc;%这一块就是在Z求左边的逆的部分的时候所用的时间
      for i_z=1:max_it_z
          z_old=z;%z就是总的目标函数中的z 110 110 100 6
          tic;
          %这里的u_Z2就是y，就应该是要表示成梯度的。
          %TV这里的z首先应该转换成梯度，然后计算
          %将110x110x6x100，然后分成两个一个x方向的一个y方向的grad
          %然后分别去进行约束（软阈值函数）
          %然后再合在一起变成y带入到z的总的式子中然后计算z
          u_Z2=ProxSparse(z+d_Z,lambda(2)/50);%此处lambda(2)是1.0是极端算子中的thea 为了保证稀疏的近端算子。 相当于y，
          d_Z=d_Z+(z-u_Z2);%d_Z是对应的admm中的lambda=lambda+z-yik+1，先对z进行约束然后再带入求Z的迭代当中
          ud_Z = fft2( u_Z2 - d_Z ) ;%ud_Z是y-lambda
          z_hat = solve_conv_term_Z(dhatT_flat, dhatTdhat_flat, ud_Z, B_hat, 50, size_z);
          z=real(ifft2(z_hat));%z变为时域上
          
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
 %显示结果
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

 













        
        
        

        

