function autoMethod = select_auto_mothod(self, inputMatrix)
    % Selects the most efficient method for non-maximum suppression automatically

    nTimes = 3;

    % Measure the time taken by each method and repeat the process multiple times
    for methodIdx = 1:4
        switch methodIdx
            case 1
                tic;
                for iT = 1:nTimes
                    if iT == 2
                        timeTic = tic;
                    end
                    [~] = self.bubble_nms(inputMatrix);
                end
                timeBubble = toc(timeTic);
            case 2
                tic;
                for iT = 1:nTimes
                    if iT == 2
                        timeTic = tic;
                    end
                    [~] = self.conv2_nms(inputMatrix);
                end
                timeConv2 = toc(timeTic);
            case 3
                tic;
                for iT = 1:nTimes
                    if iT == 2
                        timeTic = tic;
                    end
                    [~] = self.greedy_nms(inputMatrix);
                end
                timeGreedy = toc(timeTic);
            case 4
                tic;
                for iT = 1:nTimes
                    if iT == 2
                        timeTic = tic;
                    end
                    [~] = self.sort_nms(inputMatrix);
                end
                timeSort = toc(timeTic);
        end % end switch
    end % end for

    % Determine the fastest method
    [~, fastIndex] = min([timeBubble, timeConv2, timeGreedy, timeSort]);

    % Map the index to the corresponding method name
    switch fastIndex
        case 1
            autoMethod = 'bubble';
        case 2
            autoMethod = 'conv2';
        case 3
            autoMethod = 'greedy';
        case 4
            autoMethod = 'sort';
        otherwise
            error('Error: Invalid method index.');
    end

end
