%% calc the crow weights 
function weights = calc_weights(dataset)
	conv_dataset_name = ['./data/data_',dataset,'_conv5_3.mat'];
	load(conv_dataset_name);
	weights = cellfun(@(x) calc_weight(x),data,'un',0);
end
function weight = calc_weight(x)
	weight = sum(x,3);
	sumW = sum(weight(:));
	sumW = max(sumW,1e-12);
	weight = weight./sumW;
	weight = reshape(weight,[],1);
end
