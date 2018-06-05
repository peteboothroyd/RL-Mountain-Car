
%% dynamics_dp.m
% *Summary:* Implements ths ODE for simulating the triple pendulum 
% dynamics, where an input torque can be applied to each link, 
% f1:torque at inner joint, f2:torque at middle joint, f3:torque at outer joint
%
%    function dz = dynamics_dp(t, z, f1, f2, f3)
%
%
% *Input arguments:*
%
%   t     current time step (called from ODE solver)
%   z     state                                                    [6 x 1]
%   f1    (optional): torque f1(t) applied to inner pendulum
%   f2    (optional): torque f2(t) applied to middle pendulum
%   f3    (optional): torque f2(t) applied to outer pendulum
%
% *Output arguments:*
%   
%   dz    if 5 input arguments:      state derivative wrt time
%         if only 2 input arguments: total mechanical energy
%
%   Note: It is assumed that the state variables are of the following order:
%         dt1:  [rad/s] angular velocity of inner pendulum
%         dt2:  [rad/s] angular velocity of middle pendulum
%         dt3:  [rad/s] angular velocity of outer pendulum
%         t1:   [rad]   angle of inner pendulum
%         t2:   [rad]   angle of middle pendulum
%         t3:   [rad]   angle of outer pendulum
%
% A detailed derivation of the dynamics can be found in:
%
% M.P. Deisenroth: 
% Efficient Reinforcement Learning Using Gaussian Processes, Appendix C, 
% KIT Scientific Publishing, 2010.
%
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-08

function dz = dynamics_tp(t, z, f1, f2, f3)
%% Code
m1 = 0.5;  % [kg]     mass of 1st link
m2 = 0.5;  % [kg]     mass of 2nd link
m3 = 0.5;  % [kg]     mass of 3rd link
b1 = 0.0;  % [Ns/m]   coefficient of friction (1st joint)
b2 = 0.0;  % [Ns/m]   coefficient of friction (2nd joint)
b3 = 0.0;  % [Ns/m]   coefficient of friction (3rd joint)
l1 = 0.5;  % [m]      length of 1st pendulum
l2 = 0.5;  % [m]      length of 2nd pendulum
l3 = 0.5;  % [m]      length of 3rd pendulum
g  = 9.81; % [m/s^2]  acceleration of gravity
I1 = m1*l1^2/12;  % moment of inertia around pendulum midpoint (1st link)
I2 = m2*l2^2/12;  % moment of inertia around pendulum midpoint (2nd link)
I3 = m3*l3^2/12;  % moment of inertia around pendulum midpoint (2nd link)
dt1 = z(1); % current angular velocity 1
dt2 = z(2); % current angular velocity 2
dt3 = z(3); % current angular velocity 3
t1 = z(4); % current angle 1
t2 = z(5); % current angle 2
t3 = z(6); % current angle 3

if nargin == 5 % compute time derivatives

  A = [(0.25*m1+m2+m3)*l1^2+I1, l1*l2*cos(t2-t1)*(0.5*m2+m3), 0.5*m3*l1*l3*cos(t3-t1);
       l1*l2*cos(t2-t1)*(0.5*m2+m3), I2+l2^2*(0.25*m2+m3), 0.5*m3*l2*l3*cos(t3-t2);
       0.5*m3*l1*l3*cos(t3-t1), 0.5*m3*l2*l3*cos(t3-t2), 0.25*m3*l3^2+I3];
  b = [f1(t) - b1*dt1 + l1*l2*dt2*(dt2-dt1)*sin(t2-t1)*(0.5*m2+m3) + 0.5*m3*l1*l3*dt3*(dt3-dt1)*sin(t3-t1) ...
       + l1*l2*dt1*dt2*sin(t2-t1)*(0.5*m2+m3) + 0.5*m3*l1*l3*dt1*dt3*sin(t3-t1) + g*l1*sin(t1)*(0.5*m1+m2+m3);
       f2(t) - b2*dt2 + l1*l2*dt1*(dt2-dt1)*sin(t2-t1)*(m3+0.5*m2) + 0.5*m3*l2*l3*dt3*(dt3-dt2)*sin(t3-t2) ...
       - l1*l2*dt1*dt2*sin(t2-t1)*(0.5*m2+m3) + 0.5*m3*l2*l3*dt2*dt3*sin(t3-t2) + g*l2*sin(t2)*(0.5*m2+m3);
       f3(t) - b3*dt3 + 0.5*m3*(g*l3*sin(t3) - l1*l3*dt1*dt3*sin(t3-t1) - l2*l3*dt2*dt3*sin(t3-t2)) ...
       + 0.5*m3*(l1*l3*dt1*(dt3-dt1)*sin(t3-t1) + l2*l3*dt2*(dt3-dt2)*sin(t3-t2))];
  x = A\b;

  dz = zeros(6,1);
  dz(1) = x(1);
  dz(2) = x(2);
  dz(3) = x(3);
  dz(4) = dt1;
  dz(5) = dt2;
  dz(6) = dt3;

else % compute total mechanical energy
  dz = 0.125*m1*l1*l1*dt1*dt1 + 0.5*m2*(l1*l1*dt1*dt1 + l1*l2*dt1*dt2*cos(t2-t1) ...
      + 0.25*l2*l2*dt2*dt2) + 0.5*m3*(l1*l1*dt1*dt1 + 2*l1*l2*dt1*dt2*cos(t2-t1) ...
      + l1*l3*dt1*dt3*cos(t3-t1) + l2*l2*dt2*dt2 + l2*l3*dt2*dt3*cos(t3-t1) ...
      + 0.25*l3*l3*dt3*dt3) + 0.5*m1*g*cos(t1) + m2*g*(l1*cos(t1)+0.5*l2*cos(t2)) ...
      + m3*g*(l1*cos(t1)+l2*cos(t2)+0.5*l3*cos(t3)) + 0.5*(I1*I1*dt1*dt1 + I2*I2*dt2*dt2 ...
      + I3*I3*dt3*dt3);
end