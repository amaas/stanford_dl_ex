function [ output_args ] = plot_test_results( input_args )
%PLOT_TEST_RESULTS Summary of this function goes here
%   Detailed explanation goes here

heuristics_learning_rate_schedules = {'constant', 'power', 'exponential', 'halfPerEpoch'};
numHlrs= numel(heuristics_learning_rate_schedules);

learning_rate_schedules = {'none', 'adagrad', 'adadec'};
numLrs = numel(learning_rate_schedules);

adadec_forgetting_factors = [0.999 0.99 0.9];
numAff = numel(adadec_forgetting_factors);

% adadec_averaging_window_sizes = [1 2 3 5 10];
adadec_averaging_window_sizes = [1 5 10];
numAaws = numel(adadec_averaging_window_sizes);

colors = 'rgbkmcy';
shapes = 'o+*.xsd^v><ph';

figure;
hold on;

test_result_dir = 'mnist_test_results';

test_result_dir = 'svhn_test_results';

saveDir = 'svhn_test_results';


num = 0;
% for h = 2:numHlrs
    for h = 1:1
    for k = 1:numLrs
        %     for num = 3:2
        options.heuristics_learning_rate_schedule = heuristics_learning_rate_schedules{h};
        options.learning_rate_schedule = learning_rate_schedules{k};
        test_results_save_file = [saveDir '/' ...
            options.heuristics_learning_rate_schedule '_' ...
            options.learning_rate_schedule];
        if strcmp(options.learning_rate_schedule, 'adadec')
            for i = 1:numAff
                %         for i = 3:3
                options.adadec_forgetting_factor = adadec_forgetting_factors(i);
                for j = 1:numAaws
                    %             for j = 2:2
                    options.adadec_averaging_window_size = adadec_averaging_window_sizes(j);
                    options.test_results_save_file = [test_results_save_file ...
                        '_' num2str(options.adadec_forgetting_factor) ...
                        '_' num2str(options.adadec_averaging_window_size) '.mat'];
                    if exist(options.test_results_save_file, 'file')
                        load(options.test_results_save_file);
                        plot(test_results.iterations, test_results.accuracies, ['-' colors(mod(num, length(colors)) + 1) shapes(mod(num, length(shapes)) + 1)]);
                        
                        num = num + 1;
                    end
                end
            end
        else
            options.test_results_save_file = [test_results_save_file '.mat'];
            if exist(options.test_results_save_file, 'file')
                load(options.test_results_save_file);
                plot(test_results.iterations, test_results.accuracies, ['-' colors(mod(num, length(colors)) + 1) shapes(mod(num, length(shapes)) + 1)]);
                
                num = num + 1;
            end
        end
    end
end


title('Test Accuracy');
xlabel('Minibatches');
ylabel('Test accuracy (%)');
% legend('half', 'adagrad', 'adadec');
drawnow;

end

