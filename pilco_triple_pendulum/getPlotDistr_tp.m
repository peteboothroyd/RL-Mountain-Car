%% getPlotDistr_dp.m
% *Summary:* Compute means and covariances of the Cartesian coordinates of
% the tips of the inner, middle and outer pendulum assuming that the joint state
% $x$ of the cart-triple-pendulum system is Gaussian, i.e., $x\sim N(m, s)$
%
%
%     function [M1, S1, M2, S2, M3, S3] = getPlotDistr_dp(m, s, ell1, ell2, ell3)
%
%
% *Input arguments:*
%
%   m       mean of full state                                    (NOT 6) [6 x 1]
%   s       covariance of full state                              (NOT 6) [6 x 6]
%   ell1    length of inner pendulum
%   ell2    length of middle pendulum
%   ell3    length of outer pendulum
%
%   Note: this code assumes that the following order of the state:
%          1: pend1 angular velocity,
%          2: pend2 angular velocity,
%          3: pend3 angular velocity,
%          4: pend1 angle, 
%          5: pend2 angle,
%          6: pend3 angle
%
% *Output arguments:*
%
%   M1      mean of tip of inner pendulum                         [2 x 1]
%   S1      covariance of tip of inner pendulum                   [2 x 2]
%   M2      mean of tip of middle pendulum                        [2 x 1]
%   S2      covariance of tip of middle pendulum                  [2 x 2]
%   M3      mean of tip of outer pendulum                         [2 x 1]
%   S3      covariance of tip of outer pendulum                   [2 x 2]
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modification: 2013-03-27
%
%% High-Level Steps
% # Augment input distribution to complex angle representation
% # Compute means of tips of pendulums (in Cartesian coordinates)
% # Compute covariances of tips of pendulums (in Cartesian coordinates)

function [M1, S1, M2, S2, M3, S3] = getPlotDistr_tp(m, s, ell1, ell2, ell3)
%% Code

% 1. Augment input distribution
[m1, s1, c1] = gTrig(m, s, [4 5 6], [ell1, ell2, ell3]); % map input through sin/cos
m1 = [m; m1];        % mean of joint
c1 = s*c1;           % cross-covariance between input and prediction
s1 = [s c1; c1' s1]; % covariance of joint

% 2. Compute means of tips of pendulums (in Cartesian coordinates)
M1 = [-m1(7); m1(8)];                 % [-l*sin(t1), l*cos(t1)]
M2 = [-m1(7) - m1(9); m1(8) + m1(10)]; % [-l*(sin(t1)+sin(t2)),l*(cos(t1)+cos(t2))]
M3 = [-m1(7) - m1(9) - m1(11); m1(8) + m1(10) + m1(12)]; % [-l*(sin(t1)+sin(t2)+sin(t3)),l*(cos(t1)+cos(t2)+cos(t3))]

% 2. Put covariance matrices together (Cart. coord.)
% first set of coordinates (tip of 1st pendulum)
s11 = s1(7,7); 
s12 = -s1(7,8);
s22 = s1(8,8);
S1 = [s11 s12; s12' s22];

% second set of coordinates (tip of 2nd pendulum)
s11 = s1(7,7) + s1(9,9) + s1(7,9) + s1(9,7);    % ell1*sin(t1) + ell2*sin(t2)
s22 = s1(8,8) + s1(10,10) + s1(8,10) + s1(10,8);    % ell1*cos(t1) + ell2*cos(t2)
s12 = -(s1(7,8) + s1(7,10) + s1(9,8) + s1(9,10)); 
S2 = [s11 s12; s12' s22];

% third set of coordinates (tip of 3rd pendulum)
s11 = s1(7,7) + s1(7,9) + s1(7,11) + s1(9,7) + s1(9,9) + s1(9,11) ...
    + s1(11,7) + s1(11,9) + s1(11,11)  ;    % -(ell1*sin(t1) + ell2*sin(t2) + ell3*sin(t3))
s22 = s1(8,8) + s1(8,10) + s1(8,12) + s1(10,8) + s1(10,10) + s1(10,12) ...
    + s1(12,8) + s1(12,10) + s1(12,12);    % ell1*cos(t1) + ell2*cos(t2) + ell3*cos(t3)
s12 = -(s1(7,8) + s1(7,10) + s1(7,12) + s1(9,8) + s1(9,10) + s1(9,12) ...
    + s1(11,8) + s1(11,10) + s1(11,12)); 
S3 = [s11 s12; s12' s22];

% make sure we have proper covariances (sometimes numerical problems occur)
try
  chol(S1); 
catch
  warning('matrix S1 not pos.def. (getPlotDistr)');
  S1 = S1 + (1e-6 - min(eig(S1)))*eye(3);
end

try
  chol(S2); 
catch
  warning('matrix S2 not pos.def. (getPlotDistr)');
  S2 = S2 + (1e-6 - min(eig(S2)))*eye(3);
end
try
  chol(S3); 
catch
  warning('matrix S3 not pos.def. (getPlotDistr)');
  S3 = S3 + (1e-6 - min(eig(S2)))*eye(3);
end