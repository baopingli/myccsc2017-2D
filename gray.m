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
    else
        if(max(I{i}(:))>1)
            I{i}=double(I{i})/255.0;
        else
            I{i}=double(I{i});
        end
    end
end





        
        
        

        

