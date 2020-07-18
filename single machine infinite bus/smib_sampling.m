clear all; close all; clc; format compact;

% parameters
ws = 120*pi; param.ws = ws; % system angular frequency [rad/s]
Pl = 0.25; param.Pl = Pl; % initial value of load active power [pu]
M = 0.0106; param.M = M; % generator scaled inertia constant [s^2/rad]
D = 0.08; param.D = D; % generator damping constant [s/rad]
tau = 0.86; param.tau = tau; % governer time constant [s]
R = 2.5; param.R = R; % slope of machine speed-droop characteristic [pu]
Xl = 0.2; param.Xl = Xl; % line impedance [pu]
E = 1.03; param.E = E; % synchrnous generator voltage set point [pu]
Pm = 0.4; param.Pm = Pm; % generator reference active power [pu]

% random initial conditions
Nsample = 500;
x0(:,1) = -pi + (pi + pi)*rand(Nsample,1);
x0(:,2) = -10 + (10 + 10)*rand(Nsample,1);

% simulation time step
dt = 0.01;
tspan = 0 : dt : 2;

% sampling data points
X = []; Y = [];
for i1 = 1 : Nsample
    init = x0(i1,:);
    [t,x] = ode45(@(t,x) ode_smib(t,x,param),tspan,init);
    X = [X, x(1:end-1,:)'];
    Y = [Y, x(2:end,:)'];
end

% sample plot
figure; scatter(X(1,:),X(2,:));