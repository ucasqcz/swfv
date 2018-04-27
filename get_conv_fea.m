function [fea,s] = get_conv_fea(im,model,net)
    [im,s] = prepare_blob_for_cnn(im,model);
    net.blobs('data').reshape([size(im,1),size(im,2),3,1]);
    net.reshape();
    fea = net.forward({im});
end