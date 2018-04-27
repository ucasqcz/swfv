%% calc the crow weights 
function weights = calc_weights(data)
	weights = cellfun(@(x) calc_weight(x),data,'un',0);
end
function weight = calc_weight(x)
	weight = sum(x,3);
	sumW = sum(weight(:));
	sumW = max(sumW,1e-12);
	weight = weight./sumW;
	weight = reshape(weight,[],1);
end
