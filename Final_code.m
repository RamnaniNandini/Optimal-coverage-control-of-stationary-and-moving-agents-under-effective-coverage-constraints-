%% Mission Space Parameters and Agent Deployment
clc; clear all; close all;

% Grid parameters
Lx = 60;    % length in x direction
Ly = 50;    % length in y direction
dx = 1;     % grid resolution
[xg, yg] = meshgrid(0:dx:Lx, 0:dx:Ly);
points = [xg(:) yg(:)];
gridSize = [60, 50];

% Sensing model parameters
N = 9;           % number of agents to place
delta = 10;      % sensing radius
lambda = 0.12;   % decay constant
rho = 0.5;       % detection threshold
Vmax = 1;        % Max velocity (m/s)
Umax = 1;        % Max acceleration (m/s²)
T = 500;         % Total time (s)

% Probability of detection function
p_func = @(d) exp(-lambda * d) .* (d <= delta);

%% Greedy Agent Placement
agents = [];
P_current = zeros(size(points,1),1);
R = 0;
for k = 1:N
    best_H = -inf;
    best_pos = [];
    for idx = 1:size(points,1)
        candidate = points(idx,:);
        d_candidate = sqrt((points(:,1) - candidate(1)).^2 + (points(:,2) - candidate(2)).^2);
        p_candidate = p_func(d_candidate);
        P_temp = 1 - (1 - P_current) .* (1 - p_candidate);
        H = sum(P_temp);
        if H > best_H
            best_H = H;
            best_pos = candidate;
            P_best = P_temp;
        end
        
    end
    R = [R; best_H];
    agents = [agents; best_pos];
    P_current = P_best;
    fprintf('Placed agent %d at (%.2f, %.2f), coverage = %.2f\n', k, best_pos(1), best_pos(2), best_H);
end

%% Initial Coverage Visualization
P_grid = reshape(P_current, size(xg));

figure;
imagesc(0:dx:Lx, 0:dx:Ly, P_grid);
set(gca, 'YDir', 'normal');
colorbar;
title('Initial Agent Deployment Coverage');
xlabel('X position');
ylabel('Y position');
hold on;
scatter(agents(:,1), agents(:,2), 100, 'r', 'filled', 'MarkerEdgeColor','k');
legend('Agent Locations');
axis equal tight;

%% Adaptive Inspection Point Generation
[X, Y] = meshgrid(0:dx:Lx, 0:dx:Ly);
gridPoints = [X(:), Y(:)];
numPoints = size(gridPoints, 1);

%% Stationary Coverage Calculation
stationaryCoverage = ones(numPoints, 1);
for i = 1:size(agents, 1)
    d = vecnorm(gridPoints - agents(i,:), 2, 2);
    pi_x = exp(-lambda * d) .* (d <= delta);
    stationaryCoverage = stationaryCoverage .* (1 - pi_x);
end
stationaryCoverage = 1 - stationaryCoverage;

%% Step 1: Discover φM before starting inspection
% Step 1: Discover φM in Ω′ (points with coverage < rho but highest among them)
sensingRadius = delta;
omegaPrimeMask = stationaryCoverage < rho;  % Set Ω′
omegaPrimePoints = gridPoints(omegaPrimeMask, :);
omegaPrimeCoverage = stationaryCoverage(omegaPrimeMask);

[~, bestIdx] = max(omegaPrimeCoverage);  % Select max coverage within Ω′
phiM = omegaPrimePoints(bestIdx, :);     % φM from Ω′

% Update coverage using φM first
d_phiM = vecnorm(gridPoints - phiM, 2, 2);
pi_phiM = exp(-lambda * d_phiM) .* (d_phiM <= delta);
currentCoverage = 1 - (1 - pi_phiM) .* (1 - stationaryCoverage);

% Initialize inspection points and Final_H
inspectionPoints = phiM;
Final_H = mean(currentCoverage);  % Initial mean coverage performance
coveredMask = currentCoverage >= rho;

%% Adaptive Inspection Point Generation
maxIterations = 100;
samplesPerSeed = 10;
topK = 20;

for iter = 1:maxIterations
    if all(coveredMask), break; end
    
    uncoveredMask = ~coveredMask;
    uncoveredCoverage = currentCoverage(uncoveredMask);
    uncoveredPoints = gridPoints(uncoveredMask, :);

    % Get top K worst-covered points
    [~, sortIdx] = sort(uncoveredCoverage, 'ascend');
    topK_actual = min(topK, length(sortIdx));
    seeds = uncoveredPoints(sortIdx(1:topK_actual), :);

    % Sample around each seed
    candidates = [];
    for i = 1:size(seeds, 1)
        theta = 2 * pi * rand(samplesPerSeed, 1);
        r = sensingRadius * sqrt(rand(samplesPerSeed, 1));
        sampleSet = seeds(i, :) + [r .* cos(theta), r .* sin(theta)];
        sampleSet = max(sampleSet, 1);
        sampleSet = min(sampleSet, [gridSize(1), gridSize(2)]);
        candidates = [candidates; sampleSet];
    end

    % Evaluate candidates
    bestGain = -inf;
    bestSample = [];
    for s = 1:size(candidates, 1)
        d = vecnorm(gridPoints - candidates(s,:), 2, 2);
        pi_m = exp(-lambda * d) .* (d <= sensingRadius);
        newP = 1 - (1 - pi_m) .* (1 - currentCoverage);
        coverage_gain = mean(newP) - mean(currentCoverage);
        gain = coverage_gain * numPoints;

        if gain > bestGain
            bestGain = gain;
            bestSample = candidates(s,:);
        end
    end

    % Update coverage
    Final_H = [Final_H, bestGain];
    inspectionPoints = [inspectionPoints; bestSample];
    d = vecnorm(gridPoints - bestSample, 2, 2);
    pi_m = exp(-lambda * d) .* (d <= sensingRadius);
    currentCoverage = 1 - (1 - pi_m) .* (1 - currentCoverage);

    % Update masks
    coveredMask = currentCoverage >= rho;
end

%% Plot
figure; hold on; axis equal; grid on;
title('Adaptive Sampling Inspection Points');
xlabel('X [m]'); ylabel('Y [m]');
scatter(gridPoints(:,1), gridPoints(:,2), 10, currentCoverage, 'filled');
colorbar; colormap(jet); caxis([0 1]);
scatter(inspectionPoints(:,1), inspectionPoints(:,2), 50, 'k', 'filled');
scatter(agents(:,1), agents(:,2), 80, 'r', 'filled');
legend('Grid Points', 'Inspection Points', 'Stationary Agents');

%% TSP Cost Matrix with Acceleration Dynamics
nInspect = size(inspectionPoints, 1);
%Hmax = max(Final_H);
costMatrix = inf(nInspect);
Rff = transpose(R);
Hff = [Final_H, Rff];
Hmax = max(Hff);
H_bar = mean(Hff);


parfor i = 1:nInspect
    for j = 1:nInspect
        if i == j, continue; end
        
        p1 = inspectionPoints(i,:);
        p2 = inspectionPoints(j,:);
        dist = norm(p1 - p2);
        
        t_accel = Vmax/Umax;
        dist_accel = 0.5*Umax*t_accel^2;
        
        if dist > 2*dist_accel
            travelTime = (dist - 2*dist_accel)/Vmax + 2*t_accel;
        else
            travelTime = 2*sqrt(dist/Umax);
        end
        
        nSteps = 50;
        interpPts = [linspace(p1(1), p2(1), nSteps)', ...
                    linspace(p1(2), p2(2), nSteps)'];
        interpCov = arrayfun(@(k) mean(1 - (1 - exp(-lambda*...
            vecnorm(gridPoints - interpPts(k,:),2,2))) .* ...
            (1 - currentCoverage)), 1:nSteps);
        
        covCost = Hmax - mean(interpCov);
        costMatrix(i,j) = travelTime * covCost;
    end
end

%% TSP Path Construction
[~, phiM_idx] = ismember(phiM, inspectionPoints, 'rows');

tspPath = phiM_idx;
visited = false(nInspect, 1);
visited(phiM_idx) = true;

while sum(visited) < nInspect
    last = tspPath(end);
    unvisited = find(~visited);
    [~, idx] = min(costMatrix(last, unvisited));
    next = unvisited(idx);
    tspPath(end+1) = next;
    visited(next) = true;
end

tspPath(end+1) = phiM_idx;

%% Velocity Planning
segmentDist = 0;
for i = 1:(length(tspPath)-1)
    segmentDist = segmentDist + norm(inspectionPoints(tspPath(i+1), :) - ...
                                    inspectionPoints(tspPath(i), :));
end

t_accel = Vmax/Umax;
dist_accel = 0.5*Umax*t_accel^2;

if segmentDist > 2*dist_accel
    travelTime = (segmentDist - 2*dist_accel)/Vmax + 2*t_accel;
else
    travelTime = 2*sqrt(segmentDist/Umax);
end

holdTime = max(0, T - travelTime);
%H_bar = mean(Final_H);

%totalCoveragePerformance = (T * Hmax + travelTime * (H_bar - Hmax));

%% Coverage Validation
finalCoverage = currentCoverage;
uncoveredMask = finalCoverage < rho;
if any(uncoveredMask)
    error('Effective coverage not achieved!');
end

%% Final Visualization
figure; hold on;
scatter(gridPoints(:,1), gridPoints(:,2), 10, currentCoverage, 'filled');
plot(agents(:,1), agents(:,2), 'r^', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(inspectionPoints(:,1), inspectionPoints(:,2), 'bo', ...
    'MarkerSize', 10, 'LineWidth', 1.5);
plot(phiM(1), phiM(2), 'mp', 'MarkerSize', 20, 'LineWidth', 3);

for i = 1:length(tspPath)-1
    p1 = inspectionPoints(tspPath(i), :);
    p2 = inspectionPoints(tspPath(i+1), :);
    plot([p1(1), p2(1)], [p1(2), p2(2)], 'g-', 'LineWidth', 2);
end

title('Optimized Coverage Path with Acceleration Dynamics');
xlabel('X (meters)'); ylabel('Y (meters)');
legend('Coverage Map', 'Stationary Agents', 'Inspection Points', ...
    'Max Coverage Point (\phi_M)', 'TSP Path');
axis equal; colorbar; grid on;

%% Output
totalCoveragePerformance = (T * Hmax + travelTime * (H_bar - Hmax));
fprintf("Total path length: %.2f m\n", segmentDist);
fprintf("Hold time at φM: %.2f s\n", holdTime);
fprintf("Total Coverage Performance: %.2f\n", totalCoveragePerformance);
%% Animation of Agent Motion (Theorem 1 Velocity Profile)
figure;
hold on; axis equal; grid on;
title('Agent Path Animation');
xlabel('X [m]'); ylabel('Y [m]');
colormap(jet); caxis([0 1]);
scatter(gridPoints(:,1), gridPoints(:,2), 10, currentCoverage, 'filled');
plot(agents(:,1), agents(:,2), 'r^', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
plot(inspectionPoints(:,1), inspectionPoints(:,2), 'ko', 'MarkerSize', 6);
plot(phiM(1), phiM(2), 'mp', 'MarkerSize', 15, 'LineWidth', 2);
colorbar;

% Parameters
fps = 30;                     % frames per second
dt = 1/fps;
pos = inspectionPoints(tspPath(1), :);
traj = [pos];                 % to store complete path
path_segments = [];

% Construct complete trajectory (interpolated path)
for i = 1:length(tspPath)-1
    p1 = inspectionPoints(tspPath(i), :);
    p2 = inspectionPoints(tspPath(i+1), :);
    seg = [linspace(p1(1), p2(1), 100)', linspace(p1(2), p2(2), 100)'];
    path_segments = [path_segments; seg];
end

% Compute total length
distances = vecnorm(diff(path_segments), 2, 2);
total_dist = sum(distances);

% Determine velocity-time profile
t_acc = Vmax/Umax;
dist_acc = 0.5 * Umax * t_acc^2;

if total_dist > 2 * dist_acc
    t_const = (total_dist - 2 * dist_acc)/Vmax;
    total_time = 2*t_acc + t_const;
else
    t_acc = sqrt(total_dist/Umax);
    t_const = 0;
    total_time = 2*t_acc;
end

% Generate velocity profile
t = 0:dt:total_time;
v = zeros(size(t));
for i = 1:length(t)
    if t(i) <= t_acc
        v(i) = Umax * t(i);
    elseif t(i) <= t_acc + t_const
        v(i) = Vmax;
    elseif t(i) <= total_time
        v(i) = Vmax - Umax * (t(i) - t_acc - t_const);
    end
end

% Compute displacement over time
s = cumtrapz(t, v);  % cumulative distance
s = min(s, total_dist);

% Interpolate path position based on distance
cum_dist = [0; cumsum(distances)];
agent_traj = zeros(length(s), 2);
for i = 1:length(s)
    idx = find(cum_dist <= s(i), 1, 'last');
    if idx >= length(path_segments)
        agent_traj(i,:) = path_segments(end, :);
    else
        local_ratio = (s(i) - cum_dist(idx)) / distances(idx);
        agent_traj(i,:) = path_segments(idx, :) + local_ratio * ...
            (path_segments(idx+1, :) - path_segments(idx, :));
    end
end

% Animate
h = plot(agent_traj(1,1), agent_traj(1,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'y');
for i = 2:length(agent_traj)
    set(h, 'XData', agent_traj(i,1), 'YData', agent_traj(i,2));
    pause(dt);
end

% Hold at φM
holdTimeSteps = round(holdTime / dt);
for i = 1:holdTimeSteps
    set(h, 'XData', phiM(1), 'YData', phiM(2));
    pause(dt);
end

title('Agent Traversing Optimal Path and Holding at φ_M');

