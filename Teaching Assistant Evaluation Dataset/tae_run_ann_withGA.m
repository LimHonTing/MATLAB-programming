function [population, fitness_score, progress] = run_ann_withGA()

% set RNG seed number to get reproducible results; 
% change seed number to get different results
rng('default');
%seed 723, 1075, 957
rng(957);

% load dataset
data = readmatrix("tae.csv");

X = data(:, 1:end-1); % assuming all columns are attributes
Y = data(:, end); % except for the last column for labels

[~, numX] = size(X);
classY = length(unique(Y));

% Calculate the mean and standard deviation for each feature
mean_values = mean(X);
std_values = std(X);

% Normalize the features using z-score normalization
X = (X - mean_values) ./ std_values;

% what to optimize? the neural network weights
% how many values to optimize?
% again, make sure the network's input and output matches the dataset
% net = create_network(input,hidden, hidden_units_ea,output);
% num_genes = sum(net.num_parameters);

% set parameters for GA
population_size = 300; % 50 chromosomes for population
generations_max = 1000; % run for 50 generations
selrate = 0.1; % SelectionRate
mutrate = 0.3; % MutationRate
progress = [];


%initialize maximum hidden layer and hidden layer units
%seed 723, 4 hidden layer, 10 hidden layer units
%seed 1075  4 hidden layer, 10 hidden layer units
%seed 957 4 hidden layer, 10 hidden layer units
max_hidden = 4;
max_hidden_un = 10;

num_genes = (numX*max_hidden_un+max_hidden_un)+((max_hidden_un*max_hidden_un)+ ...
    max_hidden_un)*(max_hidden-1)+(max_hidden_un*classY+classY);
convergence_maxcount = 10; % stop the GA if the average fitness score stopped increasing for 5 generations
convergence_count = 0;
convergence_avg = 0;

% initialize population
population = rand(population_size, num_genes) * 2 - 1;
population = [population, randi([1,max_hidden],population_size,1), randi([1,max_hidden_un],population_size,max_hidden)];
fitness_score = zeros(population_size, 1);

% store the information for first and last generation
first_generation_population = population(:, num_genes+1:end);
last_generation_population = [];

generations_current = 1;
while generations_current < generations_max
    % test all chromosomes that haven't been tested
    for i = 1:population_size
        if fitness_score(i,1) == 0
            % fitness testing a chromosome
            fitness_score(i,1) = fitness_function(population(i, 1:end), X, Y,numX,num_genes,classY);
        end
    end

    % find out statistics of the population
    fit_avg = mean(fitness_score);

    fit_max = max(fitness_score);
    
    progress = [progress; fit_avg, fit_max];

    % convergence? 
    if fit_avg > convergence_avg
        convergence_avg = fit_avg;
        convergence_count = 0;
        disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
    else
        convergence_count = convergence_count + 1;
    end

    % perform triggered hypermutation to promote diversity and escape local
    % optima
    if (mod(generations_current, 30) == 0)
        % Define the update factors
        update_factor_mutrate = 0.1; % 10% increase
        update_factor_selrate = 0.1; % 10% decrease
        
        % Update mutation and selection rates
        mutrate = mutrate * (1 + update_factor_mutrate);
        selrate = selrate * (1 - update_factor_selrate);
        
        % Limit mutation and selection rates within a range
        mutrate = min(max(mutrate, 0), 1);
        selrate = min(max(selrate, 0), 1);
    end

    generations_current = generations_current + 1;
    % stop the GA if reach 100% accuracy or reach convergence?
    % instead of stopping immediately, slowly adjust SelRate and MutRate
    if (fit_max >= 1)
        generations_max = 0;
        last_generation_population = population(:, num_genes+1:end);
        disp("Reached convergence.")
        disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
    
    elseif (convergence_count > convergence_maxcount)
        % what to do if fitness haven't improved?
        % stop the GA?
        % generations_max = 0;
        %disp("Reached convergence.")
        % disp("Generation " + string(generations_current) + "; AvgFit " + string(fit_avg) + "; BestFit " + string(fit_max));
        
        % % or adjust selection rate and mutation rate for fine-grained search
        if (selrate < 0.95)
            ratio = fit_avg / fit_max;
            selrate = selrate + 0.1*ratio;
            mutrate = mutrate - 0.1*ratio;
        else
            generations_max = 0;
            last_generation_population = population(:, num_genes+1:end);
            disp("Reached convergence.")
        end
    end
    
    % do genetic operators
    [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate, num_genes, max_hidden, max_hidden_un, population_size);
end
    

% plot a graph of average fitness score, best fitness score vs number of
% generations
[num_prog_rows, num_prog_cols] = size(progress);
generation = 1:num_prog_rows;
avg_fit = progress(:, 1);
best_fit = progress(:, 2);
plot(generation, avg_fit, '-', generation, best_fit, '-');
xlabel('Number of Generations');
ylabel('Fitness Scores');
title('Fitness Scores VS Number of Generations');
legend('Average Fitness', 'Maximum Fitness');

% display information of first and last generation
disp("First Generation");
disp(first_generation_population);

disp("Last Generation");
disp(last_generation_population);

end


function score = fitness_function(chromosome, X, Y,input,num_genes,output)
[weight_bias_matrix,hidden_layer,hidden_lyr_unit] = splitpopulation(chromosome,num_genes);
net = create_network(input,hidden_layer, hidden_lyr_unit,output);
layers = (length(fieldnames(net))-2) / 2;

% set the weights based on the chromosome
for i = 1:layers
    layer_name = 'W' + string(i);
    bias_name = 'b' + string(i);
    
    num_genes = size(net.(layer_name), 1)*size(net.(layer_name), 2);
    new_layer = reshape(chromosome(1:num_genes), [size(net.(layer_name), 1), size(net.(layer_name), 2)]);
    net.(layer_name) = new_layer;
    chromosome = chromosome(num_genes+1:end);

    num_genes = size(net.(bias_name), 1);
    new_bias = chromosome(1:num_genes);
    net.(bias_name) = new_bias';
    chromosome = chromosome(num_genes+1:end);
end
% now test the new network
Y_pred = test(net, X);

% fitness score is the accuracy of the prediction
% what other fitness score can be calculated?
score = mean(Y == Y_pred');
end


function [population, fitness_score] = genetic_operators(population, fitness_score, selrate, mutrate, num_genes,max_hidden_layer, max_hidden_layer_unit, max_population_size)

% how many chromosomes to reject?
popsize = size(population, 1);
num_reject = round((1-selrate) * popsize);

for i = 1:num_reject
    % find lowest fitness score and remove the chromosome
    [~, lowest] = min(fitness_score);
    population(lowest, :) = [];
    fitness_score(lowest) = [];
end

% for each rejection, create a new chromosome
num_parents = size(population, 1);

for i = 1:num_reject
    % how to select parent chromosomes?
    % random permutation method
    % order = randperm(num_parents);
    order = stochastic_universal_sampling(fitness_score, 2);
    parent1 = population(order(1), :);
    parent2 = population(order(2), :);
    
    % mix-and-match
    [p1_weight_bias, p1_hidden_layer, p1_hidden_layer_units] = splitpopulation(parent1,num_genes);
    [p2_weight_bias, p2_hidden_layer, p2_hidden_layer_units] = splitpopulation(parent2,num_genes);
    if rand < 0.25
        offspring = [((p1_weight_bias + p2_weight_bias) / 2), p1_hidden_layer, p1_hidden_layer_units];
    elseif rand <0.5 
        offspring = [((p1_weight_bias + p2_weight_bias) / 2), p2_hidden_layer, p2_hidden_layer_units];
    elseif rand< 0.75
        offspring = [((p1_weight_bias + p2_weight_bias) / 2), ceil((p1_hidden_layer+p2_hidden_layer)/2), ceil((p1_hidden_layer_units+p2_hidden_layer_units)/2)];
    else
        offspring = [((p1_weight_bias + p2_weight_bias) / 2), floor((p1_hidden_layer+p2_hidden_layer)/2), floor((p1_hidden_layer_units+p2_hidden_layer_units)/2)];
    end

    % mutation
    % mut_val = rand(1, size(population(1,:), 2));
    % mut_val = mut_val * mutrate; 
    % 
    % for j = 1:size(mut_val, 2)
    %     if rand < mutrate
    %         offspring(1, j) = offspring(1, j) + mut_val(1, j);
    %     end
    % end
    
    %split population of offspring by weight and bias, hidden layer and hidden layer units
    [offspring_weight_bias, offspring_hidden_layer, offspring_hidden_layer_units] = splitpopulation(offspring,num_genes);

    % mutate on hidden layer of offspring
    if rand < mutrate
        offspring_hidden_layer = randi([1, max_hidden_layer]);
    end
    
    % mutate on hidden layer unit of offspring
    for i = 1:max_hidden_layer
        if rand < mutrate
            offspring_hidden_layer_units(i) = randi([1, max_hidden_layer_unit]);
        end
    end

    % mutate on weight and bias of offspring
    for j = 1: num_genes
        if rand < mutrate
            % add Gaussian noise with mean 0 and standard deviation 'sigma'
            sigma = 0.1;
            offspring_weight_bias(j) = offspring_weight_bias(j) + sigma * randn;
        end
    end
    offspring = [offspring_weight_bias, offspring_hidden_layer, offspring_hidden_layer_units];
    % add new offspring to population
    population = [population; offspring];
    fitness_score = [fitness_score; 0];
end

end

function [weight_bias_matrix,hidden_layer,hidden_lyr_unit] = splitpopulation(population,num_genes)
    weight_bias_matrix = population(:, 1:num_genes);
    hidden_layer = population(:, num_genes+1);
    hidden_lyr_unit = population(:, num_genes+2:end);
end

function [selected_indices] = stochastic_universal_sampling(fitness_score, numSelections)
    totalFitness = sum(fitness_score);
    pointer_distance = totalFitness / numSelections;
    start_pointer = rand * pointer_distance;

    pointers = start_pointer:pointer_distance:(totalFitness - (totalFitness - start_pointer - pointer_distance * (numSelections - 1)));

    selected_indices = zeros(1, numSelections);
    cum_fitness = cumsum(fitness_score);
    pointer_idx = 1;

    for i = 1:length(fitness_score)
        while pointer_idx <= length(pointers) && pointers(pointer_idx) <= cum_fitness(i)
            selected_indices(pointer_idx) = i;
            pointer_idx = pointer_idx + 1;
        end
    end

    selected_indices = abs(ceil(selected_indices));
    zero_indices = selected_indices == 0;
    selected_indices(zero_indices) = randi([1, length(fitness_score)]);
end
