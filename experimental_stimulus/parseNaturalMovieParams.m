function ex = parseLaneInput(varargin)
%
% FUNCTION ex = parseLaneInput(varargin)
%
% The function parseLaneInput is the template function for parsing optional input
% arguments to an experiment.
%
% (c) bnaecker@stanford.edu 20 Feb 2013	

% length of each repeat (seconds)
if any(strcmpi('time', varargin))
	ex.pa.time = varargin{find(strcmpi('time', varargin)) + 1};
else
	%ex.pa.time = 250;
    ex.pa.time = 144;
end

% number of repetitions of each stimulus block
if any(strcmpi('nreps', varargin))
	ex.pa.nReps = varargin{find(strcmpi('nreps', varargin)) + 1};
else
	ex.pa.nReps = 30;
end

% trigger
if any(strcmpi('trigger', varargin))
	ex.pa.trigger = varargin{find(strcmpi('trigger', varargin)) + 1};
else
	ex.pa.trigger = 'm';
end

% contrast for stimuli
if any(strcmpi('c', varargin))
	ex.pa.gratingContrast = varargin{find(strcmpi('c', varargin)) + 1};
	ex.pa.whiteContrast = varargin{find(strcmpi('c', varargin)) + 1};
else
	ex.pa.gratingContrast = 0.25;
	ex.pa.whiteContrast = 0.25;
end

% prepend white noise blocks
if any(strcmpi('white', varargin))
	ex.pa.useWhite = varargin{find(strcmpi('white', varargin)) + 1};
else
	ex.pa.useWhite = true;
end

if any(strcmpi('est', varargin))
    ex.pa.useEst = varargin{find(strcmpi('est', varargin)) + 1};
else
    ex.pa.useEst = true;
end

if any(strcmpi('ramp', varargin))
    ex.pa.useRamps = varargin{find(strcmpi('ramp', varargin)) + 1};
else
    ex.pa.useRamps = true;
end
