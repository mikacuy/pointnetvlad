function [SE3] = SE3MatrixFromComponents( x, y, z, r, p, yaw )
  
% SE3MatrixFromComponents - build a 4x4 matrix representing an SE(3) transform
%
% [SE3] = SE3MatrixFromComponents( x, y, z, r, p, yaw )
% 
% INPUTS:
%   x, y, z: translation
%   r, p, yaw: rotation in Euler angle representation
%
% OUTPUTS:
%   SE3: 4x4 matrix representing the SE(3) transform

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright (c) 2016 University of Oxford
% Authors: 
%  Geoff Pascoe (gmp@robots.ox.ac.uk)
%  Will Maddern (wm@robots.ox.ac.uk)
%
% This work is licensed under the Creative Commons 
% Attribution-NonCommercial-ShareAlike 4.0 International License. 
% To view a copy of this license, visit 
% http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to 
% Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  % Allow passing of a single 6-element vector
  if nargin == 1
    y = x(2);
    z = x(3);
    r = x(4);
    p = x(5);
    yaw = x(6);
    x = x(1);
  end

  % Convert euler angles to rotation matrices
  R_x = [ 
    1, 0, 0;
    0, cos(r), -sin(r);
    0, sin(r), cos(r) ];
  
  R_y = [ 
    cos(p), 0, sin(p);
    0, 1, 0;
    -sin(p), 0, cos(p) ];
  
  R_z = [
    cos(yaw), -sin(yaw), 0;
    sin(yaw), cos(yaw), 0;
    0, 0, 1];
  
  R = R_z * R_y * R_x;
  
  SE3 = [R [x; y; z]; zeros(1,3), 1];

end
