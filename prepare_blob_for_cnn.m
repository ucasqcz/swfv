function [im,im_scale] = prepare_blob_for_cnn(im,model)
    if ischar(im)
        im = imread(im);
    end
    if size(im,3) == 1
        im = repmat(im,[1,1,3]);
    end
    
    im = single(im);
    
    % load mean file
    mean_img = model.mean_img;
	if model.maxDim == 0
		im_scale = 1;
	else
		max_size = model.maxDim;
		im_scale = max_size / max(size(im));
	end
    im = imresize(im,im_scale);
    im = im - imresize(mean_img,[size(im,1),size(im,2)],'bilinear');
    
    im = im(:,:,[3,2,1]);
    im = permute(im,[2,1,3]);
    
end