function [folders,files] = split_folders_files(input)
%���������ṹ���飬����ֻ��folder�������ļ�
B=struct2cell(input) %���ṹ����ת��Ϊ��Ԫ����,��ͼƬ��������Ϣ�õ�����name folder date bytes isdir datenum
dirs=cell2mat(B(5,:)) %����Ԫ����ת��Ϊ��������logical����
folders=input(logical(dirs)) %logical����ֵת��Ϊ�߼�ֵ
files=input(~logical(dirs))%ȡ��������logical���������ͼƬ��8������6��

end

