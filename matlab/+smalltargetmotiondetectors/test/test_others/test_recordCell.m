%test_recordCell

timeList = 10000;

A = cell(100,1);
B = cell(100,1);
D = cell(100,1);
point = 100;

for timeT = 1:timeList
    C = rand(250,500);

    %%
    A(1) = [];
    A{end+1} = C;

    %%
    B = circshift(B, -1);
    B{end} = C;

    %%
    if point == 1
        point = 100;
    else
        point = point - 1;
    end
    D{point} = C;

end
