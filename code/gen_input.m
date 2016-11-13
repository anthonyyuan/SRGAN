clear;

folders = {'train','val','benchmark'};

for i=1:3
    folder = folders{i};
	if strcmp(folder,'benchmark')
		dataPath = '../dataset/benchmark';
	else
		dataPath = fullfile('../dataset/ILSVRC2015/Data/CLS-LOC/',folder);
	end
	savePath = fullfile(dataPath,'small');

	if exist(savePath,'dir')==0 
		disp(['make save path ' savePath]);
		mkdir(savePath);
	end

	dirList = dir(dataPath);
	dirList = dirList(~ismember({dirList.name},{'.','..','small'}));
	scale = 4;

	for iDir = 1:length(dirList)
		if strcmp(folder,'val')
			if iDir==1 
                imgList = dirList;
				dirList = [1]; % dummy
				dirName = '';
            else
                continue;
			end
		else
			dirName = dirList(iDir).name;
			imgList = dir(fullfile(dataPath,dirName));
			imgList = imgList(~ismember({imgList.name},{'.','..'}));
        end
        
        subDir = fullfile(savePath,dirName);
        if exist(subDir,'dir')==0 
            mkdir(subDir); 
        end
		for iImg = 1 : length(imgList)
			fileName = fullfile(dataPath,dirName,imgList(iImg).name);
			disp(sprintf('[%d/%d] %d/%d: %s',iDir,length(dirList),iImg,length(imgList),fileName));
			try
				image = imread(fileName);
			catch
				continue;
			end

			if ndims(image)==2 || (ndims(image)==3 && size(image,3)==1)
				image = cat(3,image,image,image);
			end
			image = im2double(image);

			sz = size(image);
			sz = sz(1:2);
			sz = sz-mod(sz,scale);

			target = image(1:sz(1),1:sz(2),:);
			input = imresize(target,1/scale,'bicubic');
			input = im2uint8(input);

			[path,name,ext] = fileparts(fileName);
			newName = fullfile(subDir,[name '.png']);
			
			imwrite(input,newName);
		end
	end
end
