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
    else
        if(max(I{i}(:))>1)
            I{i}=double(I{i})/255.0;
        else
            I{i}=double(I{i});
        end
    end
end





        
        
        

        

