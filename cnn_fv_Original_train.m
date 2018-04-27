function [data, fv_model, pca_model] = cnn_fv_Original_train(dataset, model, net)

    gmm_num = model.gmm_num;
    final_pca_num = model.final_pca_num;
    whiten_flag = model.whiten_flag;
    method = model.method;
    fea_name = model.fea_name;
    
    database_name = ['data_',dataset,'_,',fea_name,'_Original_',method,'_',num2str(gmm_num),'_final_pca_',num2str(final_pca_num),'_wf_',num2str(whiten_flag),'.mat'];
    database_path = fullfile('./data',database_name);
       
    gmm_name = [method,'_',dataset,'_',fea_name,'_',num2str(gmm_num),'.mat'];
    gmm_path = fullfile('./gmm_model',gmm_name);
    
    gmm_data_name = ['data_',dataset,'_',fea_name,'_','crow_',method,'_num_',num2str(gmm_num),'.mat'];
    gmm_data_path = fullfile('./data',gmm_data_name);
    
    final_pca_name = ['pca_',dataset,'_',fea_name,'_Original_',method,'_',num2str(gmm_num),'_final_pca','.mat'];
    final_pca_path = fullfile('./pca_model',final_pca_name);
    
    %% load pre pca fea feature
    fprintf('load reshape + norm conv fea\n');
    data_reshape_norm_name = ['data_',dataset,'_reshape_norm','_',fea_name,'.mat'];
    data_reshape_norm_path = fullfile('./data',data_reshape_norm_name);
    
     try
         load(data_reshape_norm_path);
     catch
     %% load conv5_3 fea
         try
            data_conv_name = ['data_',dataset,'_',fea_name,'.mat'];
            data_conv_path = fullfile('./data',data_conv_name);
            load(data_conv_path);
         catch
            data_conv_name = ['data_',dataset,'_',fea_name,'.mat'];
            data_conv_path = fullfile('./data',data_conv_name);
            data = get_dataset_conv_fea(dataset,model,net);
            save(data_conv_path,'data','-v7.3');
         end
         for idx = 1:numel(data)
             data{idx} = reshape(data{idx},[],size(data{idx},3));
             data{idx} = data{idx}';
         end
         data = cellfun(@(x) vecpostproc(x),data,'un',0);
         save(data_reshape_norm_path,'data','-v7.3');
     end
    %% load gmm model
    fprintf(['load ',method,' model\n']);
    if(strcmp(method,'gmm'))
        try
            load(gmm_path);
        catch
            fprintf(['training ',method,'---\n']);
%             data_gmm = single(cell2mat(data'));
            data = single(cell2mat(data'));
            [means,covariances,priors] = vl_gmm(data,gmm_num);
            save(gmm_path,'means','covariances','priors','-v7.3');
            clear data;
            fprintf('gmm training finished---\n');
             load(data_reshape_norm_path);
        end
    else
        try
            load(gmm_path);
        catch
             fprintf(['training ',method,'---\n']);
%             data_vlad = single(cell2mat(data'));
            data = single(cell2mat(data'));
            centers = vl_kmeans(data,gmm_num);
            kdtree = vl_kdtreebuild(centers);
            save(gmm_path,'centers','kdtree','-v7.3');
            clear data;
            load(data_reshape_norm_path);
        end
    end
    
    %% load gmm data
    if (strcmp(method,'gmm'))
        try 
            load(gmm_data_path);
        catch
            data = cellfun(@(x) vl_fisher(x,means,covariances,priors,ones(1,size(x,2)),'Improved'),data,'un',0);
        end
        fv_model.means = means;
        fv_model.covariances = covariances;
        fv_model.priors = priors;
    else
        try
             load(gmm_data_path);
        catch
             data = cellfun(@(x) vlad_cnn(x,centers,kdtree),data,'un',0);
        end
        fv_model.centers = centers;
        fv_model.kdtree = kdtree;
    end
    %% load pca model
    fprintf('load pca model\n');
    try
        final_pca_model = load(final_pca_path);
        pca_model.final_pca_model = final_pca_model;
        data = cellfun(@(x) vecpostproc(apply_whiten(x,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag)),data,'un',0);
%         save(database_path,'data','-v7.3');
    catch
        [~,eigvec,eigval,Xm] = yael_pca(single(cell2mat(data')),1024);
        final_pca_model.eigvec = eigvec;
        final_pca_model.eigval = eigval;
        final_pca_model.Xm = Xm;
        data = cellfun(@(x) vecpostproc(apply_whiten(x,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag)),data,'un',0);
        save(final_pca_path,'eigvec','eigval','Xm','-v7.3');
        pca_model.final_pca_model = final_pca_model;
%         save(database_path,'data','-v7.3');
    end
end
