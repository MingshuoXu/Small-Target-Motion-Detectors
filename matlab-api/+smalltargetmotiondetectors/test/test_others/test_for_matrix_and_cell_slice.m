times1 = 100; times2 = 10000;

m = 250; n = 500; len = 4;

C = rands(m,n);

%%
tic;
A = cell(1,1,len);
for idxI = 1:times1
    idxJ = mod(idxI,len)+1;
    A{idxJ} = C;
end
for idxI = 1:times2
    E = zeros(len,1);
    for ii = 1:len
        E(ii,1) = A{ii}(200,460);
    end
end
toc

%%
tic;
H = cell(1,1,len);
for idxI = 1:times1
    idxJ = mod(idxI,len)+1;
    H{idxJ} = C;
end

for idxI = 1:times2
    D = cell2mat(H);
    F = D(200,460,1:4);
end

toc

%%
tic;
B = zeros(m,n,len);
for idxI = 1:times1
    B(:,:,idxJ) = C;
end
for idxI = 1:times2
    F = B(200,460,1:4);
end
toc
