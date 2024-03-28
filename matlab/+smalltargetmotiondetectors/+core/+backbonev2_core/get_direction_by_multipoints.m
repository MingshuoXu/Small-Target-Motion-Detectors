function theta = get_direction_by_multipoints(points)
    %get_direction_by_multipoints
    %
    % ref: https://blog.csdn.net/qwertyu_1234567/article/details/117918602
    %

    differences = diff(points); 
    if all(differences(:) == 0)
        theta = NaN;
        return;
    end

    numPoint = size(points, 1);
    listT = 1:numPoint;
    listX = points(:,1);
    listY = points(:,2);

    sumX = sum(listX);
    sumY = sum(listY);
    sumT = sum(listT);
    sumXT = dot(listX,listT);
    SumYT = dot(listY,listT);

    veloX = numPoint*sumXT - sumX*sumT;
    veloY = numPoint*SumYT - sumY*sumT;

    %---------------------------------------%
    %   --------> y         y               %
    %   |                   ^               %
    %   |                   |               %
    %   V                   |               %
    %   x                   --------> x     %
    %                                       %
    %   matrix              image           %
    %---------------------------------------%

    % theta = atan2(-veloX, veloY);
    theta = atan2(veloY, veloX) - pi/2;

end