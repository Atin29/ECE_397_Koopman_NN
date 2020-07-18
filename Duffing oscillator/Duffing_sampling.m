clear all; close all; clc; format compact;

% parameters
param.alpha = 1;
param.beta = -1;
param.delta = 0.5;

% random initial conditions
Nsample = 500;
x0(:,1) = -1 + (1 + 1)*rand(Nsample,1);
x0(:,2) = -1 + (1 + 1)*rand(Nsample,1);

% simulation time step
dt = 0.01;
tspan = 0 : dt : 2;

% sampling data points
X = []; Y = [];
for i1 = 1 : Nsample
    init = x0(i1,:);
    [t,x] = ode45(@(t,x) ode_Duffing(t,x,param),tspan,init);
    X = [X, x(1:end-1,:)']; % paired data-set for EDMD for original state
    Y = [Y, x(2:end,:)']; % paired data-set for EDMD for original state
end

% plots
figure; scatter(X(1,:),X(2,:));