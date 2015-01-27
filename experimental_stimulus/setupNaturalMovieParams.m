function ex = setupLaneParams(ex)
%
% FUNCTION ex = setupTemplateParams(ex)
%
% The function setupTemplateParams is the template function to create experimental
% parameters.
%
% see also: RunTemplateExpt.m
%
% (c) bnaecker@stanford.edu 20 Feb 2013

%% Notify
Screen('DrawText', ex.ds.winPtr, 'Creating stimulus parameters ... ', ...
	50, 50);
Screen('Flip', ex.ds.winPtr);

%% basic information
ex.pa.experimentName = 'Lane';
ex.pa.date = datestr(now, 30);
ex.pa.saveDir = '~/Desktop';
ex = makeSaveDirectory(ex);

%% stimulus types
%ex.pa.stimType = ['white'; 'white'; cellstr(repmat('checkers', ex.pa.nReps, 1))];
%by default ex.pa.nReps = 20 (set in parseLaneInput)
%est        = cell(3, 1);
%[est{:}]   = deal('establish');
%ramps      = cell(ex.pa.nReps, 1);
%[ramps{:}] = deal('ramp');
whiteStims = cellstr(repmat('white', 5, 1));
est        = cellstr(repmat('establish', 10, 1));
ramps      = cellstr(repmat('ramp', ex.pa.nReps, 1));
asterisks  = cellstr(repmat('asterisk', ex.pa.nReps, 1));
%rampsFast  = cellstr(repmat('rampFast', ex.pa.nReps, 1));
%ex.pa.stimType = [whiteStims; est; ramps; rampsFast];
ex.pa.stimType = [whiteStims; est; ramps; asterisks];

if ~ex.pa.useWhite
    ex.pa.stimType = ex.pa.stimType(~strcmp(ex.pa.stimType,'white'));
end

if ~ex.pa.useEst
    ex.pa.stimType = ex.pa.stimType(~strcmp(ex.pa.stimType,'establish'));
end

if ~ex.pa.useRamps
    ex.pa.stimType = ex.pa.stimType(~strcmp(ex.pa.stimType,'ramp'));
end

%% aperture information
ex.pa.apertureSize = 512;					    % size of aperture, pixels
ex.pa.waitFrames = 2;						    % frames between each flip
ex.pa.umPerPix = 50 / 9;					    % approx. micron-to-pixel conversion
ex.ds.dstRect = CenterRectOnPoint(...		    % aperture destination rectangle
	[0 0 ex.pa.apertureSize ex.pa.apertureSize], ...
	ex.ds.winCtr(1), ex.ds.winCtr(2));

%% number of frames per repetition
ex.pa.nFrames = ex.pa.time * (ex.ds.frate / ex.pa.waitFrames);

%% white noise information
ex.pa.nBoxes = 32;							    % used as resolution for ALL stimuli
ex.pa.whiteContrastIndex = ones(ex.pa.nFrames, 1);

%% create conditions 
% need to have [contrasts, corrs] in conditionList
corrs = [1 -1];
contrasts = [0.05 0.35];
numLevels = 9;
ex.pa.conditionList = [(200 - 25)*(exp(linspace(0,3,numLevels)) - 1)/max(exp(linspace(0,3,numLevels))-1) + 25]';
% am now using log spacing instead of linear spacing [25.0; 50.0; 75.0; 100.0; 125.0; 150.0; 175.0; 200.0; 225.0];
ex.pa.nConditions = size(ex.pa.conditionList, 1);
middle            = median(1:ex.pa.nConditions);
if mod(ex.pa.nConditions,2) ~= 1
    error('Number of conditions not odd')
end
ex.pa.contrast    = 0.2;
ex.pa.stds        = ex.pa.contrast.*ex.pa.conditionList;
ex.pa.probes      = {-2; 0; 2};

%% random seed for each stimulus block
% Note how the seed # changes with each iteration; each sequence different
for ri = 1:length(ex.pa.stimType)
	ex.pa.random(ri).stream = ...
		RandStream.create('mrg32k3a', 'Seed', ri);
end

%% make contrasts
%ex.pa.numWhites = length(ex.pa.stimType(strcmp(ex.pa.stimType,'white')));
%ex.pa.numOthers = length(ex.pa.stimType(~strcmp(ex.pa.stimType,'white')));
%ex.pa.numMeans  = length(ex.pa.stimType(strcmp(ex.pa.stimType,'establish')));

ex.pa.sameSeqPerLevel    = cell(length(ex.pa.stimType),1);
ex.pa.whiteNoiseContrast = cell(length(ex.pa.stimType),1);
ex.pa.conditionLists  = cell(length(ex.pa.stimType),1);
ex.pa.conditions      = cell(length(ex.pa.stimType),1);
ex.pa.upOrDown        = cell(length(ex.pa.stimType),1);
ex.pa.probe           = cell(length(ex.pa.stimType),1);

%% mark all 'ramp' trials as up or down and which probe
ex.pa.rampInds = find(strcmp(ex.pa.stimType,'ramp'));
if mod(length(ex.pa.rampInds),2) ~= 0
    error(sprintf('Number of ramp stimTypes %d not divisible by 2 (up and down)', ex.pa.rampInds))
elseif mod(length(ex.pa.rampInds),3) ~=0
    error(sprintf('Number of ramp stimTypes %d not divisible by 3 (num probes)', ex.pa.rampInds))
end
temp           = repmat({'up','down'},length(ex.pa.rampInds)/2,1);
temp           = [col(temp(:)), col(repmat(ex.pa.probes, length(ex.pa.rampInds)/3,1))];
inds           = randperm(length(ex.pa.rampInds));
ex.pa.upOrDown(ex.pa.rampInds) = temp(inds,1);
ex.pa.probe(ex.pa.rampInds)    = temp(inds,2);

%% mark all 'rampFast' trials as up or down and which probe
ex.pa.rampFastInds = find(strcmp(ex.pa.stimType,'rampFast'));
if length(ex.pa.rampFastInds) > 0
    if mod(length(ex.pa.rampFastInds),2) ~= 0
        error(sprintf('Number of rampFast stimTypes %d not divisible by 2 (up and down)', ex.pa.rampFastInds))
    elseif mod(length(ex.pa.rampFastInds),3) ~=0
        error(sprintf('Number of rampFast stimTypes %d not divisible by 3 (num probes)', ex.pa.rampFastInds))
    end
    temp           = repmat({'up','down'},length(ex.pa.rampFastInds)/2,1);
    temp           = [col(temp(:)), col(repmat(ex.pa.probes, length(ex.pa.rampFastInds)/3,1))];
    inds           = randperm(length(ex.pa.rampFastInds));
    ex.pa.upOrDown(ex.pa.rampFastInds) = temp(inds,1);
    ex.pa.probe(ex.pa.rampFastInds)    = temp(inds,2);
end

%% mark all 'asterisk' trials with condition type
ex.pa.asteriskInds = find(strcmp(ex.pa.stimType,'asterisk'));
if length(ex.pa.asteriskInds) > 0
    if mod(length(ex.pa.asteriskInds),3) ~=0
        error(sprintf('Number of asterisk stimTypes %d not divisible by 3 (num probes)', ex.pa.asteriskInds))
    end
    temp           = repmat({'up','down','same'},length(ex.pa.asteriskInds)/3,1);
    temp           = [col(temp(:)), col(repmat({'up';'down';'same'}, length(ex.pa.asteriskInds)/3,1))];
    inds           = randperm(length(ex.pa.asteriskInds));
    ex.pa.upOrDown(ex.pa.asteriskInds) = temp(inds,1);
    ex.pa.probe(ex.pa.asteriskInds)    = temp(inds,2);
end


for si = 1:length(ex.pa.stimType)
    if strcmp(ex.pa.stimType(si), 'establish')
        ex.pa.conditionLists{si} = ex.pa.conditionList(randperm(ex.pa.random(si).stream, length(ex.pa.conditionList)));
        %% stretch condition list to all frames
        ex.pa.conditions{si} = imresize(ex.pa.conditionLists{si},[ex.pa.nFrames,1],'nearest');
    elseif strcmp(ex.pa.stimType(si), 'ramp')
        if strcmp(ex.pa.upOrDown(si), 'up')
            temp                     = row(ex.pa.conditionList);
            temp(middle)             = ex.pa.conditionList(middle+ex.pa.probe{si});
            ex.pa.conditionLists{si} = [ex.pa.conditionList(1), temp];
            ex.pa.conditions{si}     = imresize(ex.pa.conditionLists{si},[1,ex.pa.nFrames],'nearest');
        else
            temp                     = row(sort(ex.pa.conditionList, 'descend'));
            temp(middle)             = ex.pa.conditionList(middle+ex.pa.probe{si});
            ex.pa.conditionLists{si} = [ex.pa.conditionList(end), temp];
            ex.pa.conditions{si}     = imresize(ex.pa.conditionLists{si},[1,ex.pa.nFrames],'nearest');
        end
    elseif strcmp(ex.pa.stimType(si), 'rampFast')
        if strcmp(ex.pa.upOrDown(si), 'up')
            temp                     = row(ex.pa.conditionList);
            temp(middle)             = ex.pa.conditionList(middle+ex.pa.probe{si});
            ex.pa.conditionLists{si} = repmat([ex.pa.conditionList(1), temp], 1, 10);
            ex.pa.conditions{si}     = imresize(ex.pa.conditionLists{si},[1,ex.pa.nFrames],'nearest');
        else
            temp                     = row(sort(ex.pa.conditionList, 'descend'));
            temp(middle)             = ex.pa.conditionList(middle+ex.pa.probe{si});
            ex.pa.conditionLists{si} = repmat([ex.pa.conditionList(end), temp], 1, 10);
            ex.pa.conditions{si}     = imresize(ex.pa.conditionLists{si},[1,ex.pa.nFrames],'nearest');
        end
    elseif strcmp(ex.pa.stimType(si), 'asterisk')
        if strcmp(ex.pa.upOrDown(si), 'up')
            first                    = row(ex.pa.conditionList(1:middle));
        elseif strcmp(ex.pa.upOrDown(si), 'down')
            first                    = row(sort(ex.pa.conditionList(middle:end),'descend'));
        elseif strcmp(ex.pa.upOrDown(si), 'same')
            first                    = row(repmat(ex.pa.conditionList(middle), 1, middle));
        end
        if strcmp(ex.pa.probe(si), 'up')
            second                   = row(ex.pa.conditionList((middle+1):end));
        elseif strcmp(ex.pa.probe(si), 'down')
            second                   = row(sort(ex.pa.conditionList(1:(middle-1)),'descend'));
        elseif strcmp(ex.pa.probe(si), 'same')
            second                   = row(repmat(ex.pa.conditionList(middle), 1, middle));
        end
        ex.pa.conditionLists{si} = [first, second];
        ex.pa.conditions{si}     = imresize(ex.pa.conditionLists{si}, [1,ex.pa.nFrames],'nearest');
    end

    %% Create actual sequence of fluctuations
    if ~strcmp(ex.pa.stimType{si},'white')
        ex.pa.random(si).stateAtStimStart = ex.pa.random(si).stream.State;
        ex.pa.sameSeqPerLevel{si} = randn(ex.pa.random(si).stream, floor(ex.pa.nFrames/length(ex.pa.conditionLists{si})), 1);
        ex.pa.sameSeqPerLevel{si} = ex.pa.sameSeqPerLevel{si}/std(ex.pa.sameSeqPerLevel{si});
        ex.pa.whiteNoiseContrast{si} = repmat(col(ex.pa.sameSeqPerLevel{si}),length(ex.pa.conditionLists{si}),1);
        if length(ex.pa.whiteNoiseContrast{si}) < ex.pa.nFrames
            error(sprintf('nFrames %d not divisible by condition number %d',ex.pa.nFrames,length(ex.pa.conditionLists{si})))
        end
        ex.pa.texture{si} = col(ex.pa.contrast*ex.pa.conditions{si}).*col(ex.pa.whiteNoiseContrast{si}) + col(ex.pa.conditions{si});
        ex.pa.texture{si}(ex.pa.texture{si} > 255) = 255;
        indsOfChange = [1, row(find(diff(ex.pa.conditions{si})~=0)), ex.pa.nFrames];
        effContrasts = zeros(length(indsOfChange)-1,1);
        for i = 1:length(indsOfChange)-1
            effContrasts(i) = std(ex.pa.texture{si}(indsOfChange(i):indsOfChange(i+1)))/mean(ex.pa.texture{si}(indsOfChange(i):indsOfChange(i+1)));
        end
        ex.pa.effectiveContrast{si} = effContrasts;
    end
end


%% photodiode description
ex.pa.pdCenter = [.93 .15];                	    % center of rect, in screen percentages
ex.pa.pdRectSize = SetRect(0, 0, 100, 100);	    % actual screen size (pixels)
ex.pa.pdRect = CenterRectOnPoint(...       	    % the rectangle
    ex.pa.pdRectSize, ex.ds.winRect(3) .* ex.pa.pdCenter(1), ...
    ex.ds.winRect(4) .* ex.pa.pdCenter(2));

%% preallocate timestamp information
ex.ds.vbl = zeros(ex.pa.nFrames, ...		    % system estimate of flip time
	length(ex.pa.stimType));
ex.ds.stimOnset = ex.ds.vbl;				    % PTB's guess as to the stimulus onset
ex.ds.flipTimestamp = ex.ds.vbl;			    % timestamp after completion of flip
ex.ds.flipMissed = ex.ds.vbl;				    % PTB's guess if the flip was missed
ex.ds.beamPos = ex.ds.vbl;					    % beam position at vbl timestamp request

%% start your engines
ex.pa.initializeTime = GetSecs;
