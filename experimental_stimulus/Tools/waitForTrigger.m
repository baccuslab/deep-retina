function ex = waitForTrigger(ex)
%
% FUNCTION ex = waitForTrigger(ex)
%
% The function waitForTrigger deals with the various experiment trigger types.
% It pauses execution of the stimulus, requests that the experimenter press
% the spacebar to arm the trigger, and then either triggers on pressing the 't'
% key or uses WaitForRec, depending on the request.
%
% (c) bnaecker@stanford.edu 24 Jan 2013 

%% arm the trigger
Screen('DrawText', ex.ds.winPtr, 'Press spacebar to arm trigger ... ', ...
	50, 50);
Screen('Flip', ex.ds.winPtr);
while ~ex.kb.keyCode(ex.kb.spaceKey) && ~ex.kb.keyCode(ex.kb.escKey)
	ex = checkExptKB(ex);
end

%% wait for trigger
if any(strcmp(ex.pa.trigger, {'m', 'manual'}))
	Screen('DrawText', ex.ds.winPtr, 'Waiting for experimenter trigger (t) ... ', ...
		50, 50);
	Screen('FillOval', ex.ds.winPtr, ex.ds.black, ex.pa.pdRect);
	Screen('Flip', ex.ds.winPtr);
	while ~ex.kb.keyCode(ex.kb.tKey)
		ex = checkExptKB(ex);
	end
else
	Screen('DrawText', ex.ds.winPtr, 'Waiting for recording computer ... ', ...
		50, 50);
	Screen('FillOval', ex.ds.winPtr, ex.ds.black, ex.pa.pdRect);
	Screen('Flip', ex.ds.winPtr);
	WaitForRec;
	WaitSecs(0.5);
end

%% hide the cursor to start the experiment
HideCursor;
