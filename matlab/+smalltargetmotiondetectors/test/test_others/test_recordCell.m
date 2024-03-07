%test_recordCell

timeList = 1000;



A = cell(100,1);
B = cell(100,1);


for timeT = 1:timeList
    C = rand(250,500);

    %%
    A(1) = [];
    A{end+1} = C;

    %%
    B = circshift(B, -1);
    B{end} = C;

end
