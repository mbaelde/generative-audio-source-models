function result = compute_classification(model, test_features, test_class, method, infos)

tic
if strcmp(method, 'GMM')
    [label,~] = gmmb_classify(model, test_features);
elseif strcmp(method, 'GMM-Clavel')
    [~,proba] = gmmb_classify(model, test_features);
    winsize = 2^(nextpow2(infos.winsize * infos.fs));
    n_buff = floor((0.5 * infos.fs) / winsize);
    for i = 1:floor(size(proba,1)/n_buff)
        aux_proba = sum(log(proba(1+(i-1)*n_buff:i*n_buff,:)));
        label(1+(i-1)*n_buff:i*n_buff) = gmmb_decide(aux_proba);
    end
elseif strcmp(method, 'SVM')
    label = predict(model,test_features);
elseif strcmp(method, 'NN')
    % Test the Network
    y = model(test_features');
    [~,label] = max(y);
end

result.time_decide = toc;
if size(label,1) ~= size(test_class,1)
    label = label';
end
test_class = test_class(1:length(label));

result.label = label;
result.confusionmat = confusionmat(test_class, label);
result.confusionmatnorm =  result.confusionmat ./ repmat(sum( result.confusionmat,2),[1,size(result.confusionmat,2)]);
    
K = max(unique(test_class));
detection = zeros(K,1);
false_detection = zeros(K,1);
for k = 1:K
    group = find(test_class == k);
    false_detection(k) = sum(label(group) ~= test_class(group)) / length(test_class(group));
    detection(k) = sum(label(group) ~= test_class(group));
end
result.false_detection = false_detection;
result.false_rejection = sum(detection > 0) / K;