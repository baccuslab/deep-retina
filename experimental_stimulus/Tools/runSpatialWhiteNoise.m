function ex = runSpatialWhiteNoise(ex)
%
% FUNCTION ex = runSpatialWhiteNoise(ex)
%
% Runs a basic spatial white noise stimulus, with parameters defined in the appropriate
% way within the experimental structure 'ex'.
%
% (c) bnaecker@stanford.edu 20 Feb 2013 

%% get the stimulus block number
si = ex.pa.currentStimBlock;

%% save state of random stream
ex.pa.random(si).stateAtStimStart = ex.pa.random(si).stream.State;

%% loop over frames
vbl = GetSecs;
fi = 1;
try
while fi <= ex.pa.nFrames && ~ex.kb.keyCode(ex.kb.escKey)
	% make a spatial white noise texture
	tex = ex.ds.gray + ex.ds.gray .* ...
		ex.pa.whiteContrast(ex.pa.whiteContrastIndex(fi)) .* ...
		randn(ex.pa.random(si).stream, ex.pa.nBoxes);
	tex = min(max(tex(:), ex.ds.black), ex.ds.white);
	texture(:, :, 1) = round(reshape(tex, ex.pa.nBoxes, ex.pa.nBoxes));
	texture(:, :, 2) = ex.ds.white .* ones(ex.pa.nBoxes);
	texid = Screen('MakeTexture', ex.ds.winPtr, texture);

	% draw the texture, then kill it
	Screen('DrawTexture', ex.ds.winPtr, texid, [], ex.ds.dstRect, 0, 0);
	Screen('Close', texid);

	% draw photodiode, first box of white noise
	Screen('FillOval', ex.ds.winPtr, tex(1), ex.pa.pdRect);

	% flip the screen
	Screen('DrawingFinished', ex.ds.winPtr);
	[vbl ex.ds.stimOnset(fi, si) ex.ds.flipTimeStamp(fi, si) ...
		ex.ds.flipMissed(fi, si) ex.ds.beamPos(fi, si)] = ...
		Screen('Flip', ex.ds.winPtr, vbl + (ex.pa.waitFrames - 0.5) * ex.ds.ifi);
	
	% save the new vbl
	ex.ds.vbl(fi, si) = vbl;

	% increment frame counter
	fi = fi + 1;

	% poll keyboard
	ex = checkExptKB(ex);
end

catch me
	ex.me = me;
	return;
end

