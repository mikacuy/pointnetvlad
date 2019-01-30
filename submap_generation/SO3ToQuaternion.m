function [ q ] = SO3ToQuaternion( SO3 )

% SO3ToQuaternion - convert an SO3 matrix to quaternion representation
%
% [ q ] = SO3ToQuaternion( SO3 )
%
% INPUTS:
%   SO3: 3x3 matrix representing an SO(3) rotation
%
% OUTPUTS:
%   q: 1x4 vector representing the rotation as a quaternion ([w, x, y, z])

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

[r,c] = size( SO3 );
if( r ~= 3 || c ~= 3 )
    fprintf( 'R must be a 3x3 matrix\n\r' );
    return;
end

Rxx = SO3(1,1); Rxy = SO3(1,2); Rxz = SO3(1,3);
Ryx = SO3(2,1); Ryy = SO3(2,2); Ryz = SO3(2,3);
Rzx = SO3(3,1); Rzy = SO3(3,2); Rzz = SO3(3,3);

w = sqrt( trace( SO3 ) + 1 ) / 2;
% check if w is real. Otherwise, zero it.
if( imag( w ) > 0 )
    w = 0;
end

x = sqrt( 1 + Rxx - Ryy - Rzz ) / 2;
y = sqrt( 1 + Ryy - Rxx - Rzz ) / 2;
z = sqrt( 1 + Rzz - Ryy - Rxx ) / 2;

[~, i ] = max( [w,x,y,z] );

if( i == 1 )
    x = ( Rzy - Ryz ) / (4*w);
    y = ( Rxz - Rzx ) / (4*w);
    z = ( Ryx - Rxy ) / (4*w);
end

if( i == 2 )
    w = ( Rzy - Ryz ) / (4*x);
    y = ( Rxy + Ryx ) / (4*x);
    z = ( Rzx + Rxz ) / (4*x);
end

if( i == 3 )
    w = ( Rxz - Rzx ) / (4*y);
    x = ( Rxy + Ryx ) / (4*y);
    z = ( Ryz + Rzy ) / (4*y);
end

if( i == 4 )
    w = ( Ryx - Rxy ) / (4*z);
    x = ( Rzx + Rxz ) / (4*z);
    y = ( Ryz + Rzy ) / (4*z);
end

q = [w, x, y, z];

end
