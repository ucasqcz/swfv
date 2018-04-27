%% cnn conv fea multi-scale + fv
clear all;clc;
caffe.reset_all();
caffe.set_mode_gpu();
model_fold = './cnn_model/vgg16';
model = prepare_model(model_fold);
model.maxDim = 0;
net = caffe.Net(model.model_file,model.weights_file,'test');

img_fold = '/home1/qcz/DataSet/oxford5k/images';
gnd_train = fullfile(img_fold,'../','gnd_oxford5k.mat');
gnd_test = fullfile(img_fold,'../','gnd_oxford5k.mat');

gnd_train = load(gnd_train);
gnd_test = load(gnd_test);
try 
    load('data_oxford_conv5_3.mat');
    fprintf('load data');
catch
    img_list = gnd_train.imlist;
    data = cell(length(img_list),1);
    scale = cell(length(img_list),1);
    for ind = 1:length(img_list)
        img_path = fullfile(img_fold,[img_list{ind},'.jpg']);
        t1 = clock();
        [res,s] = get_conv_fea(img_path,model,net);
        t2 = clock();
        data{ind} = res{1};
        scale{ind} = s;
        fprintf('%d th img : %s -- time is : %f\n',ind,img_list{ind},etime(t2,t1));
    end
    save('data_oxford_conv5_3.mat','data','scale','-v7.3');
end
%% reshape + L2 normalization
fprintf('reshape + normalization\n');
% data = cellfun(@(x) reshape(x,[],size(x,3)),data,'un',0);
% data = cellfun(@(x) vecpostproc(x'),data,'un',0);
%% train fv
gmm_num = 32;
try 
    load(['gmm_',num2str(gmm_num),'.mat']);
catch
    data_gmm = single(cell2mat(data'));
    clear data;
    fprintf('trainging gmm ---\n');
    [means,covariances,priors] = vl_gmm(data_gmm,gmm_num);
    save(['gmm_',num2str(gmm_num),'.mat'],'means','covariances','priors','-v7.3');
    fprintf('gmm training finished---\n');
end
try 
    load(['data_fv_',num2str(gmm_num),'.mat']);
catch
    data_fv = cellfun(@(x) vl_fisher(x,means,covariances,priors),data,'un',0);
    save(['data_fv_',num2str(gmm_num),'.mat'],'data_fv','-v7.3');
end
%% learning pca 
try
    load('pca_32.mat');
catch
    [~,eigvec,eigval,Xm] = yael_pca(single(cell2mat(data_fv')));
end
%% apply pca-whitening
fprintf('applying pca-whiterning\n');
try
    load('data_pca_512_32.mat');
catch
    pca_dim = 512;  
    whiten_flag = 1;
    data_pca = cellfun(@(x) vecpostproc(apply_whiten(x,Xm,eigvec,eigval,pca_dim,whiten_flag)),data_fv,'un',0);
end
%% query images
fprintf('process query images\n');
qim_list = {gnd_test.imlist{gnd_test.qidx}};
qims = arrayfun(@(x) crop_qim(fullfile(img_fold,[qim_list{x},'.jpg']),gnd_test.gnd(x).bbx),1:numel(gnd_test.qidx),'un',0);
q_fea = cell(length(qims),1);
for i = 1:length(qims)
    [res,~] = get_conv_fea(qims{i},model,maxDim,net);
    q_fea{i} = res{1};
end
% normalization
fprintf('reshape + normalization query \n');
q_fea = cellfun(@(x) reshape(x,[],size(x,3)),q_fea,'un',0);
q_fea = cellfun(@(x) vecpostproc(x'),q_fea,'un',0);
% fv feature
fprintf('generate fv feature query \n');
q_fea_fv = cellfun(@(x) vl_fisher(x,means,covariances,priors),q_fea,'un',0);
clear q_fea;
% pca + whitening
fprintf('pca + whitening query \n');
pca_dim = 512; 
whiten_flag = 1;
q_fea_pca = cellfun(@(x) vecpostproc(apply_whiten(x,Xm,eigvec,eigval,pca_dim,whiten_flag)),q_fea_fv,'un',0);
clear q_fea_fv;
vecs = cell2mat(data_pca');
qvecs = cell2mat(q_fea_pca');
[ranks,sim] = yael_nn(vecs,-qvecs,size(vecs,2),16);
map = compute_map(ranks,gnd_test.gnd);
