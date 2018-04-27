function [out] = fisher_combine(x,means,covariances,priors,varargin)
    if nargin == 4
        weight = ones(1,size(x,2));
        ratio = 1;
    elseif nargin == 5
        weight = varargin{1};
        ratio = 1;
    elseif nargin == 6
        weight = varargin{1};
        ratio = varargin{2};
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
	x = x(:,ind);
	weight = weight(ind);
	
    out = vl_fisher(x,means,covariances,priors,weight,'Improved');  
end