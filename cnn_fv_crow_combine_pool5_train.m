function [data, fv_model, pca_model] = cnn_fv_crow_combine_pool5_train(dataset, model, net)

    gmm_num = model.gmm_num;
    final_pca_num = model.final_pca_num;
    whiten_flag = model.whiten_flag;
    
    if model.combine_gmm
        if strcmp(dataset,'oxford5k')
            dataset_compen = 'paris6k';
        elseif strcmp(dataset,'paris6k')
            dataset_compen = 'oxford5k';
        end 
    end
    if model.combine_gmm && model.cross_pca
        final_pca_name = ['pca_','com_',num2str(model.combine_gmm),'_',dataset_compen,'_pool5_gmm_',num2str(gmm_num),'_final_pca_crow','.mat'];
        final_pca_path = fullfile('./pca_model',final_pca_name);
    else
        final_pca_name = ['pca_','com_',num2str(model.combine_gmm),'_',dataset,'_pool5_gmm_',num2str(gmm_num),'_final_pca_crow','.mat'];
        final_pca_path = fullfile('./pca_model',final_pca_name);  
    end
    database_name = ['data_',dataset,'_pool5_gmm_',num2str(gmm_num),'_final_pca_',num2str(final_pca_num),'_wf_',num2str(whiten_flag),'.mat'];
    database_path = fullfile('./data',database_name);
    if ~exist('./data','dir') mkdir('./data');end
       
    gmm_name = ['gmm_','com_',num2str(model.combine_gmm),'_',dataset,'_pool5_',num2str(gmm_num),'.mat'];
    gmm_path = fullfile('./gmm_model',gmm_name);
    if ~exist('./gmm_model','dir') mkdir('./gmm_model');end
    
%     final_pca_name = ['pca_','com_',num2str(model.combine_gmm),'_',dataset,'_pool5_gmm_',num2str(gmm_num),'_final_pca_crow','.mat'];
%     final_pca_path = fullfile('./pca_model',final_pca_name);
    if ~exist('./pca_model','dir') mkdir('./pca_model');end
    
    weight_file = [dataset,'_pool5_weights.mat'];
    weight_path = fullfile('./data',weight_file);

    data_pool_name = ['data_',dataset,'_pool5','.mat'];
    data_pool_path = fullfile('./data',data_pool_name);
    
    gmm_data_name = ['data_','com_',num2str(model.combine_gmm),'_',dataset,'_pool5_crow_gmm_num_',num2str(gmm_num),'.mat'];
    gmm_data_path = fullfile('./data',gmm_data_name);
  
    %% calc weights
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
    data_reshape_norm_name = ['data_',dataset,'_reshape_norm','_pool5','.mat'];
    data_reshape_norm_path = fullfile('./data',data_reshape_norm_name);
    
    data_compen_reshape_norm_name = ['data_',dataset_compen,'_reshape_norm','_pool5','.mat'];
    data_compen_reshape_norm_path = fullfile('./data',data_compen_reshape_norm_name);
    
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
    fprintf('load gmm model\n');
    try
        load(gmm_path);
%         data = cellfun(@(x) vl_fisher(x,means,covariances,priors,'Improved'),data,'un',0);
%         data = cellfun(@(x,y) fisher_combine(x,means,covariances,priors,y),data,weights,'un',0);
    catch
         if model.combine_gmm
          %% load dataset_compen
             data_compen = load(data_compen_reshape_norm_path);
             data_compen = data_compen.data;
             data_gmm = single(cell2mat([data;data_compen]'));
             clear data_compen;
         else
            data_gmm = single(cell2mat(data'));
         end
        fprintf('training gmm ---\n');
        [means,covariances,priors] = vl_gmm(data_gmm,gmm_num);
        clear data_gmm;
        save(gmm_path,'means','covariances','priors','-v7.3');
%         data = cellfun(@(x) vl_fisher(x,means,covariances,priors,'Improved'),data,'un',0);
        fprintf('gmm training finished---\n');
    end
	try 
		load(gmm_data_path);
	catch
		data = cellfun(@(x,y) fisher_combine(x,means,covariances,priors,y,1),data,weights,'un',0);
        save(gmm_data_path,'data','-v7.3');
	end
    fv_model.means = means;
    fv_model.covariances = covariances;
    fv_model.priors = priors;
    %% load pca model
    fprintf('load pca model\n');
    
    data = cell2mat(data');
    try
        load(final_pca_path);
        final_pca_model.eigvec = eigvec;
        final_pca_model.eigval = eigval;
        final_pca_model.Xm = Xm;
        pca_model.final_pca_model = final_pca_model;
    catch
        [~,eigvec,eigval,Xm] = yael_pca(single(data));
        final_pca_model.eigvec = eigvec(:,1:4096);
        final_pca_model.eigval = eigval(1:4096,:);
        
        final_pca_model.Xm = Xm;
        pca_model.final_pca_model = final_pca_model;
        save(final_pca_path,'eigvec','eigval','Xm','-v7.3');
    end
    data = vecpostproc(apply_whiten(data,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag));
%     try
%         final_pca_model = load(final_pca_path);
%         pca_model.final_pca_model = final_pca_model;
%         data = cellfun(@(x) vecpostproc(apply_whiten(x,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag)),data,'un',0);
%         save(database_path,'data','-v7.3');
%     catch
%         [~,eigvec,eigval,Xm] = yael_pca(single(cell2mat(data')));
%         final_pca_model.eigvec = eigvec;
%         final_pca_model.eigval = eigval;
%         final_pca_model.Xm = Xm;
%         data = cellfun(@(x) vecpostproc(apply_whiten(x,final_pca_model.Xm,final_pca_model.eigvec,final_pca_model.eigval,final_pca_num,whiten_flag)),data,'un',0);
%         save(final_pca_path,'eigvec','eigval','Xm','-v7.3');
%         pca_model.final_pca_model = final_pca_model;
%         save(database_path,'data','-v7.3');
%     end
    
    %% load database
%     try
%         data = load(database_path);
%         s_fieldnames = fieldnames(data);
%         data = data.(s_fieldnames{1});
%     catch
%         if ~exist('data_fv','var')
%             data_fv = cellfun(@(x) vl_fisher(x,means,covariances,priors,'Improved'),data,'un',0);
%         end
%         data = cellfun(@(x) vecpostproc(apply_whiten(x,Xm,eigvec,eigval,pca_num,whiten_flag)),data_fv,'un',0);
%         save(database_path,'data','-v7.3');
%     end
end