function [ output_args ] = train_sweep_parameters( input_args )
%TRAIN_SWEEP_PARAMETERS Summary of this function goes here
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

options.test_interval_iterations = 10;
options.numFigures = 1;

saveDir = 'svhn_test_results';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

for h = 1:numHlrs
    % for h = 1:1
    for num = 1:1
        %     for num = 3:3
        options.heuristics_learning_rate_schedule = heuristics_learning_rate_schedules{h};
        options.learning_rate_schedule = learning_rate_schedules{num};
        test_results_save_file = [saveDir '/' ...
            options.heuristics_learning_rate_schedule '_' ...
            options.learning_rate_schedule];
        if strcmp(options.learning_rate_schedule, 'adadec')
            for i = 1:numAff
                options.adadec_forgetting_factor = adadec_forgetting_factors(i);
                for j = 1:numAaws
                    options.adadec_averaging_window_size = adadec_averaging_window_sizes(j);
                    options.test_results_save_file = [test_results_save_file ...
                        '_' num2str(options.adadec_forgetting_factor) ...
                        '_' num2str(options.adadec_averaging_window_size) '.mat'];
                    cnnTrain(options);
                end
            end
        else
            options.test_results_save_file = [test_results_save_file '.mat'];
            cnnTrain(options);
        end
    end
end


end

