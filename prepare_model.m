function model = prepare_model(model_fold)
    model.mean_file = fullfile(model_fold,'mean_image.mat');
    model.model_file = fullfile(model_fold,'test.prototxt');
    model.weights_file = fullfile(model_fold,'model.caffemodel');
    model.img_fold = '/home1/qcz/DataSet';    


    s = load(model.mean_file);
    s_fieldnames = fieldnames(s);
    assert(length(s_fieldnames) == 1);
    model.mean_img = s.(s_fieldnames{1});
end
