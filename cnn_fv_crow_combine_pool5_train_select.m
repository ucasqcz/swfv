function [data, fv_model, pca_model] = cnn_fv_crow_combine_pool5_train_select(dataset, model, net)

    gmm_num = model.gmm_num;
    final_pca_num = model.final_pca_num;
    whiten_flag = model.whiten_flag;
    fea_name = model.fea_name;
    
    
    database_name = ['data_',dataset,'_',fea_name,'_',model.method,'_',num2str(gmm_num),'_final_pca_',num2str(final_pca_num),'_select_',num2str(model.select),'_wf_',num2str(whiten_flag),'.mat'];
    database_path = fullfile('./data',database_name);
       
    gmm_name = [model.method,'_',dataset,'_',fea_name,'_',num2str(gmm_num),'.mat'];
    gmm_path = fullfile('./gmm_model',gmm_name);
    
    final_pca_name = ['pca_',dataset,'_',fea_name,'_',model.method,'_',num2str(gmm_num),'_final_pca_crow_select_',num2str(model.select),'.mat'];
    final_pca_path = fullfile('./pca_model',final_pca_name);
    
    weight_file = [dataset,'_',fea_name,'_','weights.mat'];
    weight_path = fullfile('./data',weight_file);

    data_pool_name = ['data_',dataset,'_',fea_name,'.mat'];
    data_pool_path = fullfile('./data',data_pool_name);
    
    gmm_data_name = ['data_',dataset,'_',fea_name,'_','crow_',num2str(model.select),'_',model.method,'_num_',num2str(gmm_num),'.mat'];
    gmm_data_path = fullfile('./data',gmm_data_name);
    
    try
        load(weight_path);
    catch
         try
            load(data_pool_path);
         catch
            data = get_dataset_conv_fea(dataset,model,net);
            save(data_pool_path,'data','-v7.3');
         end
        weights = calc_weights(data);
        save(weight_path,'weights','-v7.3');
        clear data;
    end
    %% load pre pca fea feature
    fprintf('load reshape + norm conv fea\n');
    data_reshape_norm_name = ['data_',dataset,'_reshape_norm','_',fea_name,'.mat'];
    data_reshape_norm_path = fullfile('./data',data_reshape_norm_name);
    
     try
         load(data_reshape_norm_path);
     catch
     %% load pool5 fea
         try
            load(data_pool_path);
         catch
            data = get_dataset_conv_fea(dataset,model,net);
            save(data_pool_path,'data','-v7.3');
         end
         for idx = 1:numel(data)
             data{idx} = reshape(data{idx},[],size(data{idx},3));
             data{idx} = data{idx}';
         end
         data = cellfun(@(x) vecpostproc(x),data,'un',0);
         save(data_reshape_norm_path,'data','-v7.3');
     end
    %% load gmm model
    fprintf(['load ',model.method,' model\n']);
    if strcmp(model.method,'gmm')
        try
            load(gmm_path);
        catch
            data_gmm = single(cell2mat(data'));
            fprintf('training gmm ---\n');
            [means,covariances,priors] = vl_gmm(data_gmm,gmm_num);
            clear data_gmm;
            save(gmm_path,'means','covariances','priors','-v7.3');
    %         data = cellfun(@(x) vl_fisher(x,means,covariances,priors,'Improved'),data,'un',0);
            fprintf('gmm training finished---\n');
        end
    else
        try
            load(gmm_path);
        catch
            data_vlad = single(cell2mat(data'));
            fprintf(['training ',model.method,'---\n']);
            centers = vl_kmeans(data_vlad,gmm_num);
            kdtree = vl_kdtreebuild(centers);
            save(gmm_path,'centers','kdtree','-v7.3');
            clear data_vlad;
        end
    end
    if strcmp(model.method,'gmm')
        try 
            load(gmm_data_path);
        catch
            data = cellfun(@(x,y) fisher_combine(x,means,covariances,priors,y,model.select),data,weights,'un',0);
            save(gmm_data_path,'data','-v7.3');
        end
        fv_model.means = means;
        fv_model.covariances = covariances;
        fv_model.priors = priors;
    else
         try
             load(gmm_data_path);
        catch
             data = cellfun(@(x,y) vlad_cnn(x,centers,kdtree,y,model.select),data,weights,'un',0);
        end
        fv_model.centers = centers;
        fv_model.kdtree = kdtree;
    end
    %% load pca model
    fprintf('load pca model\n');
    try
        final_pca_model = load(final_pca_path);
        pca_model.final_pca_model = final_pca_model;
		data = cell2mat(data');
		data = vecpostproc(apply_whiten(data,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag));
        % data = cellfun(@(x) vecpostproc(apply_whiten(x,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag)),data,'un',0);
%         save(database_path,'data','-v7.3');
    catch
        [~,eigvec,eigval,Xm] = yael_pca(single(cell2mat(data')),1024);
        final_pca_model.eigvec = eigvec;
        final_pca_model.eigval = eigval;
        final_pca_model.Xm = Xm;
        % data = cellfun(@(x) vecpostproc(apply_whiten(x,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag)),data,'un',0);
		data = cell2mat(data');
		data = vecpostproc(apply_whiten(data,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag));
        save(final_pca_path,'eigvec','eigval','Xm','-v7.3');
        pca_model.final_pca_model = final_pca_model;
%         save(database_path,'data','-v7.3');
    end
end