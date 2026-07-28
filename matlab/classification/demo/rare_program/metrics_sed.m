function [f1_score, error_rate] = metrics_sed(matrix_true, matrix_pred)

[n_class, N] = size(matrix_true);

true_positive = zeros(1,N);
false_positive = zeros(1,N);
false_negative = zeros(1,N);
substitution = zeros(1,N);
deletion = zeros(1,N);
insertion = zeros(1,N);
n_active = zeros(1,N);
for n = 1:N
    true = matrix_true(:,n);
    pred = matrix_pred(:,n);
    for k = 1:n_class
        if true(k) == 1 && pred(k) == 1
            true_positive(n) = true_positive(n) + 1;
        elseif true(k) == 1 && pred(k) == 0
            false_negative(n) = false_negative(n) + 1;
        elseif true(k) == 0 && pred(k) == 1;
            false_positive(n) = false_positive(n) + 1;
        end
    end
    substitution(n) = min(false_negative(n),false_positive(n));
    deletion(n) = max([0, false_negative(n) - false_positive(n)]);
    insertion(n) = max([0, false_positive(n) - false_negative(n)]);
    
    n_active(n) = sum(matrix_true(:,n));
end
precision = sum(true_positive) ./ (sum(true_positive) + sum(false_positive));
recall = sum(true_positive) ./ (sum(true_positive) + sum(false_negative));

%f1_score = 2 * precision * recall ./ (precision + recall);
f1_score = 2 * sum(true_positive) ./ ( 2 * sum(true_positive) + sum(false_positive) + sum(false_negative) );
error_rate = (sum(substitution) + sum(deletion) + sum(insertion)) ./ sum(n_active);