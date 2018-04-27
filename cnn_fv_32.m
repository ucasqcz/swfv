%% cnn conv fea multi-scale + fv
clear all;clc;
caffe.reset_all();
caffe.set_mode_gpu();
model_fold = './model/vgg16';
model = prepare_model(model_fold);

net = caffe.Net(model.model_file,model.weights_file,'test');

img_fold = 'R:\qcz\Dataset\Oxford5k\oxbuild_images';
gnd_train = fullfile(img_fold,'../','gnd_oxford5k.mat');
gnd_test = fullfile(img_fold,'../','gnd_oxford5k.mat');

gnd_train = load(gnd_train);
gnd_test = load(gnd_test);
try 
    load('data_oxford_conv5_3.mat');
catch
    img_list = gnd_train.imlist;
    data = cell(length(img_list),1);
    for ind = 1:length(img_list)
        img_path = fullfile(img_fold,[img_list{ind},'.jpg']);
        [im,scale] = prepare_blob_for_cnn(img_path,model,600);
        t1 = clock;
        net.blobs('data').reshape([size(im,1),size(im,2),3,1]);
        net.reshape();
        res = net.forward({im});
        t2 = clock();
        data{ind} = res{1};
        fprintf('%d th img : %s -- time is : %f\n',ind,img_list{ind},etime(t2,t1));
    end
    save('data_oxford_conv5_3.mat','data','-v7.3');
end
%% reshape + L2 normalization
fprintf('reshape + normalization\n');
data = cellfun(@(x) reshape(x,[],size(x,3)),data,'un',0);
data = cellfun(@(x) vecpostproc(x'),data,'un',0);
%% train fv
gmm_num = 256;
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


