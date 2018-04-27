function enc =  vlad_cnn(data,centers,kdtree,varargin)

    if nargin < 4
        weight = ones(1,size(data,2));
        ratio =  1;
    elseif  nargin == 4
        weight = varargin{1};
        ratio = 1;
    elseif nargin == 5
        weight = varargin{1};
        ratio  = varargin{2};
    end
    
    if size(weight,2) ~= 1
		weight = weight';
	end
    % cut the small 
    weight_sort = sort(weight,'descend');
    weight_sum = cumsum(weight_sort);
    thre = 0;
    for i = 1:numel(weight)
        if weight_sum(i) > ratio
            thre = weight_sort(i);
            break;
        end
    end
    weight(weight<thre) = 0;
	%% not compute non-select
	ind = find(weight ~= 0);
	data=data(:,ind);
	weight = weight(ind);
    
    
    
    nn = vl_kdtreequery(kdtree,centers,data);
    numClusters = size(centers,2);
    numData = size(data,2);
    assignments = zeros(numClusters,numData,'single');
    assignments(sub2ind(size(assignments),nn,1:length(nn))) = 1;
    
    enc = vl_vlad(data,centers,assignments,weight,'SquareRoot');
end