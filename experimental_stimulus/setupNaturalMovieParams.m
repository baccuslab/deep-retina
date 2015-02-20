function ex = setupNaturalMovieParams(ex)
%
% FUNCTION ex = setupNaturalMovieParams(ex)
%
% The function setupNaturalMovieParams is the function to create experimental
% parameters.
%

%% Notify
Screen('DrawText', ex.ds.winPtr, 'Creating stimulus parameters ... ', ...
	50, 50);
Screen('Flip', ex.ds.winPtr);

%% basic information
ex.pa.experimentName = 'NaturalMovieStimulus';
ex.pa.date = datestr(now, 30);
ex.pa.saveDir = '~/Desktop';
ex = makeSaveDirectory(ex);

whiteStims = cellstr(repmat('white', 5, 1));
natural    = cellstr(repmat('natural', ex.pa.nReps, 1));
ex.pa.stimType = [whiteStims; natural];

if ~ex.pa.useWhite
    ex.pa.stimType = ex.pa.stimType(~strcmp(ex.pa.stimType,'white'));
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
% store a cell array of the image names to ex.pa.imgNames
imageStruct = dir(ex.pa.imgDir);
ex.pa.imgNames = cell(length(imageStruct));
for img = 1:length(ex.pa.imgNames)
    if ~isempty(findstr('LUM',imageStruct(img).name))
        ex.pa.imgNames{img} = imageStruct(img).name;
    end
end
ex.pa.imgNames = ex.pa.imgNames(~cellfun('isempty',ex.pa.imgNames));

% store a cell array of the image paths
ex.pa.imgPaths = cell(length(ex.pa.imgNames));
for img = 1:length(ex.pa.imgPaths)
    ex.pa.imgPaths{img} = strcat(ex.pa.imgDir, '/', ex.pa.imgNames{img});
end

%% random seed for each stimulus block
% Note how the seed # changes with each iteration; each sequence different
for ri = 1:length(ex.pa.stimType)
	ex.pa.random(ri).stream = ...
		RandStream.create('mrg32k3a', 'Seed', ri);
end

%% make contrasts


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
