function generate_submaps(base_path)
    %To Use: generate_submaps('/media/deep-three/deep_ssd2/Robotcar/2014-05-19-13-05-38')
  
    %%%%%%%%%%%%Folder Locations%%%%%%%%
    %lidar
    base_path= strcat(base_path, '/');
    laser='lms_front';
    laser_dir= strcat(base_path,laser,'/');
    pc_output_folder='pointclouds/';
    
    %make pc output folder
    mkdir(strcat(base_path,pc_output_folder));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%Load extrinsics%%%%%%%%
    extrinsics_dir='/home/vgd/Documents/robotcar-dataset-sdk-2.1/extrinsics/';
    laser_extrinisics = dlmread([extrinsics_dir 'lms_front.txt']);
    ins_extrinsics = dlmread([extrinsics_dir 'ins.txt']);

    G_ins_laser = SE3MatrixFromComponents(ins_extrinsics) \ SE3MatrixFromComponents(laser_extrinisics);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%Timestamps%%%%%%%%%%%%%
    laser_timestamps = dlmread(strcat(base_path,'/',laser,'.timestamps'));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%Parameters%%%%%%%%%%%%%%%
    to_display=1;
    
    start_chunk=1;
    target_pc_size=4096;
    
    %submap generation
    submap_cover_distance=20.0;
    laser_reading_distance=0.025;
    laser_reading_angle=30;
    dist_start_next_frame=10.0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%Set up CSV file%%%%%%%%%%%%
    csv_file_name= 'test.csv';
    fid_locations=fopen(strcat(base_path,csv_file_name), 'w');
    fprintf(fid_locations,'%s,%s,%s\n','timestamp','northing','easting');
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    for chunk = start_chunk:laser_timestamps(end,2)
        %find readings in chunk
        laser_index_start= find(laser_timestamps(:,2) == chunk, 1, 'first');
        laser_index_end= find(laser_timestamps(:,2) == chunk, 1, 'last');

        l_timestamps=laser_timestamps(laser_index_start:laser_index_end,1);

        disp(strcat('Processing chunk: ',num2str(chunk),' Laser Start Index: ',num2str(laser_index_start),' Laser End Index: ',num2str(laser_index_end)));
        %filter edge cases
        if (chunk==1)
           %remove first few readings (in car park)
           l_timestamps=laser_timestamps(laser_index_start+5000:laser_index_end,1);
        end

        if (chunk==laser_timestamps(end,2))
           %remove last readings
           l_timestamps=laser_timestamps(laser_index_start:laser_index_end-1000,1);
        end

        %%%%%%%%%%POSES%%%%%%%%%%
        laser_global_poses=getGlobalPoses(strcat(base_path,'/gps/ins.csv'), l_timestamps');
        disp(strcat('Processing chunk: ',num2str(chunk),' Loaded laser poses'));
        %%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%Counter Variables%%%%
        %laser
        frame_start=1;
        frame_end=frame_start+1;
        frames=[];
        i=frame_start;
        j=i;
        start_next_frame=frame_start;
        got_next=0;
        %%%%%%%%%%%%%%%%%%%%%%%%%%

        while(frame_end<length(l_timestamps))
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%GET SCANS TO GENERATE SUBMAP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            while(getDistance(laser_global_poses{i}(1,4), laser_global_poses{i}(2,4),laser_global_poses{frame_start}(1,4), laser_global_poses{frame_start}(2,4))<submap_cover_distance)
                if(j>(length(l_timestamps)-1))
                   break
                end  
                j=j+1;  

                while((getDistance(laser_global_poses{i}(1,4), laser_global_poses{i}(2,4), laser_global_poses{j}(1,4), laser_global_poses{j}(2,4))<laser_reading_distance)...
                       && (getRotation(laser_global_poses{i}(1:3,1:3), laser_global_poses{j}(1:3,1:3))*180/pi <laser_reading_angle))
                    j=j+1;
                    if(j>(length(l_timestamps)-1))
                        break
                    end  
                end
                frames=[frames j];

                if(j>(length(l_timestamps)-1))
                    break
                end

                if(getDistance(laser_global_poses{frame_start}(1,4), laser_global_poses{frame_start}(2,4), laser_global_poses{j}(1,4), laser_global_poses{j}(2,4))>dist_start_next_frame && got_next==0)
                  start_next_frame=frames(1,end);
                  got_next=1;
                end
            i=j;
            end

            if(j>length(l_timestamps)-1)
                break
            end  
            frame_end=j;        
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%%%%%Build Pointcloud%%%%%%%
            pointcloud = [];
            for i=frames
                scan_path = [laser_dir num2str(l_timestamps(i,1)) '.bin'];
                scan_file = fopen(scan_path);
                scan = fread(scan_file, 'double');
                fclose(scan_file);

                scan = reshape(scan, [3 numel(scan)/3]);
                scan(3,:) = zeros(1, size(scan,2));

                scan = inv(laser_global_poses{frame_start})*laser_global_poses{i} * G_ins_laser * [scan; ones(1, size(scan,2))];
                pointcloud = [pointcloud scan(1:3,:)];
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            %%%Remove ground plane
            [normal, in_plane, out_plane]=pcfitplane(pointCloud(pointcloud'),0.5);

            %%%%%%%%%%%Check if not enough points after road removal
            if (size(out_plane,1)<target_pc_size)
                %reset variables
                if (got_next==0)
                   frame_start=frame_start+50;
                   start_next_frame=frame_start+7;
                else
                   frame_start=start_next_frame;
                   start_next_frame=frame_start;
                end
                frame_end= frame_start+1;
                frames=[frame_start];
                i=frame_start;
                j=i;               
                got_next=0;

                disp('Faulty pointcloud');
                continue 
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%Downsample to exactly target_pc_size points%%%%%%%%%%%%
            out_of_plane=pointcloud(:,out_plane);
            
            %find appropriate scale
            scale_size=1.001;
            downsampled=pcdownsample(pointCloud(out_of_plane'),'gridAverage',scale_size);
            
            while (downsampled.Count()<target_pc_size)
               scale_size=scale_size-0.025;
               if(scale_size<=0)
                    xyz=out_of_plane';
                    break;
               end
               downsampled=pcdownsample(pointCloud(out_of_plane'),'gridAverage',scale_size);
            end
            
            while (downsampled.Count()>target_pc_size)
               scale_size=scale_size+0.025;
               downsampled=pcdownsample(pointCloud(out_of_plane'),'gridAverage',scale_size);
            end
            
            if(scale_size>0)
                xyz=[downsampled.Location(:,1),downsampled.Location(:,2),downsampled.Location(:,3)];
            end 
            
            %add additional random points
            num_extra_points=target_pc_size-size(xyz,1);
            permutation=randperm(length(out_of_plane));
            sample_out=permutation(1:num_extra_points);
            sample=out_of_plane(:,sample_out);%3xn            
            
            output=[xyz',sample];            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%%%%%%%%%%%%%%%Transform pointcloud%%%%%%%%%%%%%%%%%%%%%%%%%%
            %transform wrt the centroid
            x_cen=mean(output(1,:));
            y_cen=mean(output(2,:));
            z_cen=mean(output(3,:));
            centroid=[x_cen;y_cen;z_cen;1];
            centroid_g=double(laser_global_poses{frame_start})*double(centroid);
            
            %make spread s=0.5/d
            sum=0;
            for i=1:size(output,2)
                sum=sum+sqrt((output(1,i)-x_cen)^2+(output(2,i)-y_cen)^2+(output(3,i)-z_cen)^2);
            end
            d=sum/size(output,2);
            s=0.5/d;

            T=[[s,0,0,-s*(x_cen)];...
            [0,s,0,-s*(y_cen)];...
            [0,0,s,-s*(z_cen)];...
            [0,0,0,1]];
            scaled_output=T*[output; ones(1, size(output,2))];
            scaled_output=-scaled_output;
            
            %Enforce to be in [-1,1] and have exactly target_pc_size points
            cleaned=[];
            for i=1:size(scaled_output,2)
               if(scaled_output(1,i)>=-1 && scaled_output(1,i)<=1 && scaled_output(2,i)>=-1 && scaled_output(2,i)<=1 ...
                       && scaled_output(3,i)>=-1 && scaled_output(3,i)<=1)
                    cleaned=[cleaned,scaled_output(:,i)];
               end
            end
            
            %make number of points equal to target_pc_size
            num_extra_points=target_pc_size-size(cleaned,2);
            disp(strcat(num2str(size(cleaned,2)),'.',num2str(num_extra_points)));
            permutation=randperm(length(out_of_plane));
            i=1;
            while size(cleaned,2)<target_pc_size
               new_point=-T*[out_of_plane(:,permutation(1,i));1];
               if(new_point(1,1)>=-1 && new_point(1,1)<=1 && new_point(2,1)>=-1 && new_point(2,1)<=1 ...
                       && new_point(3,1)>=-1 && new_point(3,1)<=1)                
                    cleaned=[cleaned,new_point];
               end
               i=i+1;
            end
            cleaned=cleaned(1:3,:);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            %%%Double check
            if(size(cleaned,2)~=target_pc_size)
               frame_start=start_next_frame;
               frame_end= frame_start+1;
               frames=[frame_start];
               i=frame_start;
               j=i;
               disp('Invalid pointcloud')
               continue;
            end
            
            %%%Output Files
            %output pointcloud in binary file
            origin_timestamp=l_timestamps(frames(1,1),1);
            fileID = fopen(strcat(base_path,pc_output_folder, num2str(origin_timestamp),'.bin'),'w');
            fwrite(fileID,cleaned,'double');
            fclose(fileID);
            disp(num2str(origin_timestamp));
            
            %write line in csv file
            fprintf(fid_locations, '%s,%f,%f\n',num2str(origin_timestamp),centroid_g(1,1), centroid_g(2,1));
            
            %%%Display
            if(to_display)
                figure(1);
                pcshow(cleaned');
                axis equal;
                pause
            end
            
            %%%%%%%Reset Variables%%%%%%
            if (got_next==0)
               frame_start=frame_start+50;
            else
               frame_start=start_next_frame;
            end
            frame_end= frame_start+1;
            frames=[frame_start];
            i=frame_start;
            j=i;               
            got_next=0;    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        end
    end
    fclose(fid_locations);
    plot_pointcloud_path(base_path);
end
