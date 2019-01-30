function plot_pointcloud_path(base_path)
    %base_path='/media/deep-two/896cd795-f10d-4e67-b8c0-69bbf9c978f1/Robotcar/2014-12-05-11-09-10/';
    data = importdata(strcat(base_path,'pointcloud_locations_20m.csv'));
    full_path=[data.data];
    
%     data=[data.data]; 
%     diff=[];
%     for i= 1:size(data,1)-1
%         diff=[diff,getDistance(data(i,2), data(i,3), data(i+1,2), data(i+1,3))];
%     end
    
%     hist(diff);
    
%     base_path2='/media/deep-two/deep_ssd/Robotcar/2015-11-13-10-28-08/';
%     data2 = importdata(strcat(base_path2,'pointcloud_locations_20m.csv'));
%     data=[data2.data];
    
    x_width=150;
    y_width=150;
    
    %validation region for NUS
%     p1=[363621.292362,142864.19756];%51
%     p2=[364788.795462,143125.746609];
%     p3=[363597.507711,144011.414174];
    
    %validation region for West Coast
%     p1=[360895.486453,144999.915143];%41
%     p2=[362357.024536,144894.825301];%393
%     p3=[361368.907155,145209.663042];    

%for oxford
p1=[5735712.768124,620084.402381];
p2=[5735611.299219,620540.270327];
p3=[5735237.358209,620543.094379];
p4=[5734749.303802,619932.693364];  

    p=[p1;p2;p3;p4];
    
    x_ranges=[];
    y_ranges=[];
    for i=1:size(p,1)
        x_ranges=[x_ranges; [p(i,1)-x_width,p(i,1)+x_width];[p(i,1)-x_width,p(i,1)+x_width]];
        y_ranges=[y_ranges; [p(i,2)-y_width,p(i,2)+y_width];[p(i,2)-y_width,p(i,2)+y_width]];
    end    
    
%     x_ranges=[[p1(1,1)-x_width,p1(1,1)+x_width];[p2(1,1)-x_width,p2(1,1)+x_width];[p3(1,1)-x_width,p3(1,1)+x_width];[p4(1,1)-x_width,p4(1,1)+x_width]];
%     y_ranges=[[p1(1,2)-y_width, p1(1,2)+y_width];[p2(1,2)-y_width, p2(1,2)+y_width];[p3(1,2)-y_width, p3(1,2)+y_width];[p4(1,1)-x_width,p4(1,1)+x_width]];
%     x_ranges=[[p1(1,1)-x_width,p1(1,1)+x_width];[p3(1,1)-x_width,p3(1,1)+x_width]];
%     y_ranges=[[p1(1,2)-y_width, p1(1,2)+y_width];[p3(1,2)-y_width, p3(1,2)+y_width]];    
%     x_threshold= 5.7352*10^6;
%     y_threshold= 6.201*10^5;

%     x_threshold=5.73565*10^6;
%     y_threshold=6.2029*10^5;

    train=[];
    test=[];
    count=0;
    for i=1:size(full_path,1)
      %if full_path(i,2)<x_threshold && full_path(i,3)<y_threshold
      test_set=0;
      for j=1:size(x_ranges,1)
          if x_ranges(j,1)<full_path(i,2) && full_path(i,2)<x_ranges(j,2) && y_ranges(j,1)<full_path(i,3) && full_path(i,3)<y_ranges(j,2)
             test_set=1;
             count=count+1;
             break;
          end    
      end
      if test_set
         test=[test;full_path(i,:)];
      else
          train=[train;full_path(i,:)];
      end  
    end
    %disp(size(train,1));
    %disp(size(test,1));
    %disp(data1.data(1,2));
    %plot(global_poses(1,4:4:end),global_poses(2,4:4:end),'b');
    %hold on;
    %plot(data(2:end,2),data(2:end,3),'r.');
    figure('name',num2str(count));
    
%     plot(data(:,2),data(:,3),'g.');
    
    plot(train(:,2),train(:,3),'b.');
    hold on;
    plot(test(:,2),test(:,3),'r.');
    hold on;
    plot(p1(1,1),p1(1,2),'o');
    hold on;
    rectangle('Position',[p1(1,1)-x_width, p1(1,2)-y_width, x_width*2, y_width*2],'EdgeColor','k')
    hold on;    
    
    plot(p2(1,1),p2(1,2),'x');
    hold on;
    rectangle('Position',[p2(1,1)-x_width, p2(1,2)-y_width, x_width*2, y_width*2],'EdgeColor','k')
    hold on;
    
    plot(p3(1,1),p3(1,2),'*');
    hold on;
    rectangle('Position',[p3(1,1)-x_width, p3(1,2)-y_width, x_width*2, y_width*2],'EdgeColor','k')
    hold on;
    
    plot(p4(1,1),p4(1,2),'*');
    hold on;
    rectangle('Position',[p4(1,1)-x_width, p4(1,2)-y_width, x_width*2, y_width*2],'EdgeColor','k')
    
    axis equal;
end