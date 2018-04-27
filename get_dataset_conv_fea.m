function data = get_dataset_conv_fea(dataset,model,net)
    %img_fold = ['R:\qcz\Dataset\',dataset,'\images'];
    fold = model.img_fold;
    img_fold = fullfile(fold,dataset,'images');
    
    gnd_train = fullfile(img_fold,'../',['gnd_',dataset,'.mat']);
    gnd_train = load(gnd_train);

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
end
