
clc,clear,close all;

%%

for m = [200,250,500]
    for n = [200,250,500]
        for c = 0.1:0.1:0.9

            for t = 1:100
                A = rand(m,n)-c;


                %%
                opt1 = half_wave_rectification(A);
                opt2 = max(A,0);


                %%
                opt4 = half_wave_rectification(-A);
                opt5 = max(-A,0);


            end
            if max(abs(opt1-opt2),[],'all') || ...
                    max(abs(opt4-opt5),[],'all') 
                lastwarn(sprintf('!!!!!(%d,%d,%.1f\n)',m,n,c));
            else
                fprintf('(%d,%d,%.1f)\n',m,n,c);
            end

        end
    end
end

function Y = half_wave_rectification(X)
    %Half_Wave_Rectification
    Y = (X+abs(X)) / 2;
    %the fastest approch: max(X,0) and -min(X,0)
end
