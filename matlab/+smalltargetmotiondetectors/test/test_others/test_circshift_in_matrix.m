times = 10000;

m = 4; n = 100; 

A = rands(m,n);
B = A; D = A; A1 = A;
C = rands(m,1);
%%
tic;
for idxI = 1:times
    A(:,1) = [];
    A(:,end+1) = C;

end
toc

tic;
for idxI = 1:times
    A1(:,end+1) = C;
end
toc

tic;
for idxI = 1:times
    A1(:,1) = [];
end
toc

tic;
for idxI = 1:times
    B = [B(:,2:end),C];
end
toc

tic;
for idxI = 1:times
    D = circshift(D, -1);
    D(:,end) = C;
end
toc
