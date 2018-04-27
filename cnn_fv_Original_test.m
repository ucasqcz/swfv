%% test
%% cnn conv fea multi-scale + fv
clear all;clc;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(2);
model_fold = './cnn_model/vgg16';
model = prepare_model(model_fold);
model.fea_name = 'ft_pool5';
fea_name = model.fea_name;

net = caffe.Net(model.model_file,model.weights_file,'test');

dataset = 'paris6k';
model.method = 'vlad';
% model.method = 'gmm';

g_n = [128,256];
p_n = [128,256,512,1024];
w_f = [1,0];
for g_i = 1:2
for p_i = 1:4
for w_i = 1:2
    
model.maxDim = 0;
model.gmm_num = g_n(g_i);
model.final_pca_num = p_n(p_i);
model.whiten_flag = w_f(w_i);
txt_name = ['result_Original_',model.method,'_,',fea_name,'_',dataset,'.txt'];
result_file = fopen(txt_name,'a+');

img_fold = ['/home1/qcz/DataSet/',dataset,'/images'];
gnd_train = fullfile(img_fold,'../',['gnd_',dataset,'.mat']);
gnd_test = gnd_train;

gnd_train = load(gnd_train);
gnd_test = load(gnd_test);
[data,fv_model,pca_model] = cnn_fv_Original_train(dataset,model,net);

%% query images
fprintf('process query images\n');
qim_list = {gnd_test.imlist{gnd_test.qidx}};
q_fea_name = ['data_query_',dataset,'_',fea_name,'.mat'];
q_fea_path = fullfile('./data',q_fea_name);
try
    q_fea = load(q_fea_path);
    s_fieldnames = fieldnames(q_fea);
    q_fea = q_fea.(s_fieldnames{1});
catch
    qims = arrayfun(@(x) crop_qim(fullfile(img_fold,[qim_list{x},'.jpg']),gnd_test.gnd(x).bbx),1:numel(gnd_test.qidx),'un',0);
    q_fea = cell(length(qims),1);
    for i = 1:length(qims)
        [res,~] = get_conv_fea(qims{i},model,net);
        q_fea{i} = res{1};
    end
    save(q_fea_path,'q_fea','-v7.3');
end
% normalization
fprintf('reshape + pca query \n');
q_fea = cellfun(@(x) reshape(x,[],size(x,3)),q_fea,'un',0);
q_fea = cellfun(@(x) vecpostproc(x'),q_fea,'un',0);
% fv feature
fprintf(['generate ',model.method,' feature query \n']);
if strcmp(model.method,'gmm')
    q_fea_fv = cellfun(@(x) vl_fisher(x,fv_model.means,fv_model.covariances,fv_model.priors,ones(1,size(x,2)),'Improved'),q_fea,'un',0);
    clear q_fea;
else
    q_fea_fv = cellfun(@(x) vlad_cnn(x,fv_model.centers,fv_model.kdtree),q_fea,'un',0);
end
% pca + whitening
fprintf('pca + whitening query \n');
final_pca_model = pca_model.final_pca_model;
q_fea_pca = cellfun(@(x) vecpostproc(apply_whiten(x,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,model.final_pca_num,model.whiten_flag)),q_fea_fv,'un',0);
clear q_fea_fv;
data = cell2mat(data');
qvecs = cell2mat(q_fea_pca');
[ranks,sim] = yael_nn(data,-qvecs,size(data,2),16);
map = compute_map(ranks,gnd_test.gnd);

fprintf(result_file,'%s: %d final_pca: %d whiten: %d result : %f\n',model.method,model.gmm_num,model.final_pca_num,model.whiten_flag,map);
fprintf('%s: %d final_pca: %d whiten: %d result : %f\n',model.method,model.gmm_num,model.final_pca_num,model.whiten_flag,map);
fclose(result_file);
end
end
end