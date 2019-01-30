function theta=getRotation(R1,R2)
    [x1, y1, z1]= getEulerAngles(R1);
    [x2, y2, z2]= getEulerAngles(R2);
    
    theta=sqrt((x1-x2)^2+(y1-y2)^2+(z1-z2)^2);
end