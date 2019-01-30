function [T_matrices] = getGlobalPoses(ins_file, pose_timestamps)
  
% InterpolatePoses - interpolate INS poses to find poses at given timestamps
%
% [poses] = InterpolatePoses(ins_file, pose_timestamps, origin_timestamp)
%
% INPUTS:s
%   ins_file: csv file containing INS data
%   pose_timestamps: array of UNIX timestamps at which interpolated poses are 
%     required
%   origin_timestamp: timestamp for origin frame, relative to which poses are
%     reported
%
% OUTPUTS:
%   poses: cell array of 4x4 matrices, representing SE3 poses at the times 
%     specified in pose_timestamps

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

  ins_file_id = fopen(ins_file);
  headers = textscan(ins_file_id, '%s', 15, 'Delimiter',',');
  ins_data = textscan(ins_file_id, ...
      '%u64 %s %f %f %f %f %f %f %s %f %f %f %f %f %f','Delimiter',',');
  fclose(ins_file_id);
  
  ins_timestamps = ins_data{1};
  
  lower_index = max(find(ins_timestamps(:,1)<=min(pose_timestamps), 1, ...
      'last')-1,1);
  upper_index = find(ins_timestamps(:,1)>max(pose_timestamps), 1, 'first')+1;
  
  lower_index = max(lower_index, 1);
  upper_index = min(upper_index, numel(ins_timestamps));
  
  ins_poses = cell(1, upper_index - lower_index + 1);
  ins_quaternions = cell(1, upper_index - lower_index + 1);
  
  northings = ins_data{6};
  eastings = ins_data{7};
  downs = ins_data{8};
  rolls = ins_data{13};
  pitches = ins_data{14};
  yaws = ins_data{15};
  
  
  for i=1:upper_index
    ins_poses{i} = SE3MatrixFromComponents(...
      northings(i), eastings(i), downs(i), rolls(i), pitches(i), yaws(i));
    ins_quaternions{i} = SO3ToQuaternion(ins_poses{i}(1:3,1:3))';
  end
  
  [lower_index_rows, lower_index_cols] = ...
    find(and(bsxfun(@le,ins_timestamps,pose_timestamps)',...
    circshift(bsxfun(@gt,ins_timestamps,pose_timestamps)',[0 -1])));

  lower_indices = zeros(size(lower_index_rows));
  lower_indices(lower_index_rows) = lower_index_cols;
  lower_indices = max(lower_indices, 1);
  
  ins_timestamps = cast(ins_timestamps, 'double');
  pose_timestamps = cast(pose_timestamps, 'double');
  fractions = (pose_timestamps - ins_timestamps(lower_indices)')./...
    (ins_timestamps(lower_indices+1)'-ins_timestamps(lower_indices)');
  
  quaternions_lower = [ins_quaternions{lower_indices}];
  quaternions_upper = [ins_quaternions{lower_indices+1}];
  
  % interpolate quaternions
  d_array = sum(quaternions_lower.*quaternions_upper,1);
  
  linear_interp_indices = find(~(abs(d_array)<1.0));
  sin_interp_indices = find(abs(d_array)<1.0);
  
  scale0_array = zeros(size(d_array));
  scale1_array = zeros(size(d_array));
  
  scale0_array(linear_interp_indices) = 1-fractions(linear_interp_indices);
  scale1_array(linear_interp_indices) = fractions(linear_interp_indices);
  
  theta_array = acos(abs(d_array(sin_interp_indices)));
  
  scale0_array(sin_interp_indices) = sin((1-fractions(sin_interp_indices)).*...
      theta_array)./sin(theta_array);
  scale1_array(sin_interp_indices) = sin(fractions(sin_interp_indices).*...
      theta_array)./sin(theta_array);
    
  negative_d_indices = find(d_array < 0);
  scale1_array(negative_d_indices) = -scale1_array(negative_d_indices);
  
  quaternions_interp = repmat(scale0_array,4,1).*quaternions_lower + ...
      repmat(scale1_array,4,1).*quaternions_upper;
    
  % interpolate positions
  ins_poses_array = [ins_poses{:}];
  positions_lower = ins_poses_array(1:3,4*lower_indices);
  positions_upper = ins_poses_array(1:3,4*(lower_indices+1));
  positions_interp = repmat((1-fractions),3,1).*positions_lower + ...
      repmat(fractions,3,1).*positions_upper;
  
  poses_array = zeros(4, 4*numel(pose_timestamps));
  poses_array(1,1:4:end) = 1-2*quaternions_interp(3,:).^2 ...
      -2*quaternions_interp(4,:).^2;
  poses_array(1,2:4:end) = 2*quaternions_interp(2,:).*...
    quaternions_interp(3,:) - 2*quaternions_interp(4,:).*...
    quaternions_interp(1,:);
  poses_array(1,3:4:end) = 2*quaternions_interp(2,:).*...
      quaternions_interp(4,:) + 2*quaternions_interp(3,:).*...
      quaternions_interp(1,:);

  poses_array(2,1:4:end) = 2*quaternions_interp(2,:).*...
      quaternions_interp(3,:) + 2*quaternions_interp(4,:).*...
      quaternions_interp(1,:);
  poses_array(2,2:4:end) = 1-2*quaternions_interp(2,:).^2 ...
      -2*quaternions_interp(4,:).^2;
  poses_array(2,3:4:end) = 2*quaternions_interp(3,:).*...
      quaternions_interp(4,:) - 2*quaternions_interp(2,:).*...
      quaternions_interp(1,:);

  poses_array(3,1:4:end) = 2*quaternions_interp(2,:).*...
      quaternions_interp(4,:) - 2*quaternions_interp(3,:).*...
      quaternions_interp(1,:);
  poses_array(3,2:4:end) = 2*quaternions_interp(3,:).*...
      quaternions_interp(4,:) + 2*quaternions_interp(2,:).*...
      quaternions_interp(1,:);
  poses_array(3,3:4:end) = 1-2*quaternions_interp(2,:).^2 ...
      -2*quaternions_interp(3,:).^2;
  
  poses_array(1:3,4:4:end) = positions_interp;
  poses_array(4,4:4:end) = 1; %poses_array --> concatinated 4x4 matrices wrt global
  
  %%%
  global_poses=poses_array;
  %%%
  T_matrices = cell(numel(pose_timestamps),1);
  for i=1:numel(pose_timestamps)
    T_matrices{i} = poses_array(:,(i-1)*4+1:(i)*4);
  end

end
