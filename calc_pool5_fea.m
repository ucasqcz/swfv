clear all;clc;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(3);
model_fold = './cnn_model/vgg16';
model = prepare_model(model_fold);
model.model_file = fullfile(model_fold,'pool5.prototxt');
net = caffe.Net(model.model_file,'test');

dataset_names = {'oxford5k','paris6k'};
for i = 1:2
dataset = dataset_names{i};
model.img_fold = ['/data1/NLPRMNT/wanghongsong/qcz/ObjectRetrieval/Rmac/datasets/',dataset,'/images'];
model.maxDim = 0;

data_conv_name = ['data_query_',dataset,'_conv5_3','.mat'];
data_conv_path = fullfile('./data',data_conv_name);


load(data_conv_path);
data = q_fea;
for index = 1:length(data)
	tmp = data{index};
	net.blobs('data').reshape([size(tmp,1),size(tmp,2),size(tmp,3),1]);
	net.reshape();
	tmp = net.forward({tmp});
	data{index} = tmp{1};
	fprintf('pool5 fea for %d th img---\n',index);
end
data_pool5_name = ['data_query_',dataset,'_pool5','.mat'];
data_pool5_path = fullfile('./data',data_pool5_name);

save(data_pool5_path,'data','-v7.3');
end