function result = benchmark_classification(input)

database = input.database;

descriptors = input.descriptors;
infos = input.infos;
tic
[training_features,training_class, test_features, test_class] = compute_descriptors(database, descriptors, infos);
elapsed_descriptor = toc;
method = input.method;

disp('Create model...')
tic
if strcmp(method(1:3), 'GMM')
    model = gmmb_create(training_features, training_class, 'GEM');
elseif strcmp(method(1:3),'SVM')
    model = fitcecoc(training_features, training_class,'Coding','onevsall','verbose',2);
elseif strcmp(method(1:2), 'NN')
    model = fitnn(training_features, training_class);
end
elapsed_model = toc;

result = compute_classification(model, test_features, test_class, method, infos);

result.time_descriptor_training = elapsed_descriptor;
result.time_model_training = elapsed_model;