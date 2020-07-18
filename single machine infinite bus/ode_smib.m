function F = ode_smib(t,x,param)
% parameters
ws = param.ws;
M = param.M;
D = param.D;
tau = param.tau;
R = param.R;
Xl = param.Xl;
E = param.E;
Pm = param.Pm;
Pl = param.Pl;

% differential equations for SMIB dynamics
F = zeros(2,1);
F(1) = x(2); % angle
F(2) = 1/M*(Pm - E/Xl*sin(x(1)) - Pl - D*x(2)); % frequency