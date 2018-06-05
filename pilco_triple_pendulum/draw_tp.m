%% draw_dp.m
% *Summary:* Draw the triple-pendulum system with reward, applied torques, 
% and predictive uncertainty of the tips of the pendulums
%
%    function draw_dp(theta1, theta2, theta3, f1, f2, f3, cost, text1, text2, M, S)
%
% *Input arguments:*
%
%   theta1     angle of inner pendulum
%   theta2     angle of middle pendulum
%   theta3     angle of outer pendulum
%   f1         torque applied to inner pendulum
%   f2         torque applied to middle pendulum
%   f3         torque applied to outer pendulum
%   cost       cost structure
%     .fcn     function handle (it is assumed to use saturating cost)
%     .<>      other fields that are passed to cost
%   text1      (optional) text field 1
%   text2      (optional) text field 2
%   M          (optional) mean of state
%   S          (optional) covariance of state
%
%
% Copyright (C) 2008-2013 by
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2013-03-07

function draw_tp(theta1, theta2, theta3, f1, f2, f3, cost, text1, text2, M, S)
%% Code
l = 0.5;
xmin = -3*l; ymin = -3*l;
xmax = 3*l;  ymax= 3*l; 
umax = 2;
height = 0;

% Draw double pendulum
clf; hold on
sth1 = sin(theta1); sth2 = sin(theta2); sth3 = sin(theta3);
cth1 = cos(theta1); cth2 = cos(theta2); cth3 = cos(theta3);
pendulum1 = [0, 0; 
             -l*sth1, l*cth1];
pendulum2 = [-l*sth1, l*cth1; 
             -l*(sth1+sth2), l*(cth1+cth2)];
pendulum3 = [-l*(sth1+sth2), l*(cth1+cth2); 
             -l*(sth1+sth2+sth3), l*(cth1+cth2+cth3)];
         
plot(pendulum1(:,1),pendulum1(:,2),'r','linewidth',4)
plot(pendulum2(:,1),pendulum2(:,2),'r','linewidth',4)
plot(pendulum3(:,1),pendulum3(:,2),'r','linewidth',4)

% plot target location
plot(0,3*l,'k+','MarkerSize',20);
plot([xmin, xmax], [-height, -height],'k','linewidth',2)
% plot inner joint
plot(0,0,'k.','markersize',24)
plot(0,0,'y.','markersize',14)
% plot middle joint
plot(-l*sth1, l*cth1,'k.','markersize',24)
plot(-l*sth1, l*cth1,'y.','markersize',14)
% plot outer joint
plot(-l*(sth1+sth2), l*(cth1+cth2),'k.','markersize',24)
plot(-l*(sth1+sth2), l*(cth1+cth2),'y.','markersize',14)
% plot tip of outer joint
plot(-l*(sth1+sth2+sth3), l*(cth1+cth2+cth3),'k.','markersize',24)
plot(-l*(sth1+sth2+sth3), l*(cth1+cth2+cth3),'y.','markersize',14)
plot(0,-3*l,'.w','markersize',0.005)

% % Draw sample positions of the joints
% if nargin > 7
%   samples = gaussian(M,S+1e-8*eye(4),1000);
%   t1 = samples(3,:); t2 = samples(4,:);
%   plot(-l*sin(t1),l*cos(t1),'b.','markersize',2)
%   plot(-l*(sin(t1)-sin(t2)),l*(cos(t1)+cos(t2)),'r.','markersize',2)
% end

% plot ellipses around tips of pendulums (if M, S exist)
if exist('S','var') && max(max(S))>0
    [M1, S1, M2, S2, M3, S3] = getPlotDistr_tp(M, S, l, l, l);
    error_ellipse(S1,M1,'style','b'); % inner pendulum
    error_ellipse(S2,M2,'style','r'); % middle pendulum
    error_ellipse(S3,M3,'style','g'); % outer pendulum
end
% try
%   
% catch
%   
% end

% Show other useful information
% plot applied torques
plot([0 f1/umax*xmax],[-0.3, -0.3],'g','linewidth',10)
plot([0 f2/umax*xmax],[-0.5, -0.5],'g','linewidth',10)
plot([0 f3/umax*xmax],[-0.7, -0.7],'g','linewidth',10)
% plot reward
reward = 1-cost.fcn(cost,[0, 0, 0, theta1, theta2, theta3]',zeros(6));
plot([0 reward*xmax],[-0.9, -0.9],'y','linewidth',10)
text(0,-0.3,'applied  torque (inner joint)')
text(0,-0.5,'applied  torque (middle joint)')
text(0,-0.7,'applied  torque (outer joint)')
text(0,-0.9,'immediate reward')
if exist('text1','var')  
  text(0,-1.1, text1)
end
if exist('text2','var')
  text(0,-1.3, text2)
end
set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[ymin ymax]);
axis off
drawnow;