function [folders,files] = split_folders_files(input)
%返回两个结构数组，其中只有folder和输入文件
B=struct2cell(input) %将结构数组转换为单元数组,将图片的所有信息得到包括name folder date bytes isdir datenum
dirs=cell2mat(B(5,:)) %将单元数组转换为单个矩阵logical数组
folders=input(logical(dirs)) %logical将数值转换为逻辑值
files=input(~logical(dirs))%取出非正的logical的数组就是图片从8个到了6个

end

