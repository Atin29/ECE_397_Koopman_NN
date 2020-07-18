function F = ode_Duffing(t,x,param)
% --------------------------------------
% ODE parameters
% --------------------------------------
alpha = param.alpha;
beta = param.beta;
delta = param.delta;

% --------------------------------------
% Dynamic States
% --------------------------------------
x1 = x(1); % x
x2 = x(2); % diff(x,1)

% --------------------------------------
% Duffing oscillator dynamics
% --------------------------------------
F = zeros(2,1);
F(1) = x2;
F(2) = -delta*x2 - x1*(beta + alpha*x1^2);