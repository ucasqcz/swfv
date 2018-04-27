% apply PCA-whitening, with or without dimensionality reduction
function x_ = apply_whiten (x, xm, eigvec, eigval, dout, whiten_flag)

if ~exist ('dout')
  dout = size (x, 1);
end

x_ = bsxfun (@minus, x, xm);  % Subtract the mean
if whiten_flag
    x_ = diag(eigval(1:dout).^-0.5)*eigvec(:,1:dout)' * x_;
else
    x_ = eigvec(:,1:dout)' * x_;
end

x_ = replacenan (x_);

% replace all nan values in a matrix (with zero)
function y = replacenan (x, v)

if ~exist ('v')
  v = 0;
end

y = x;
y(isnan(x)) = v;	