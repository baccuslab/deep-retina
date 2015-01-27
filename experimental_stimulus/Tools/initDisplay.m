function ex = initDisplay(ex)
%
% FUNCTION ex = initDisplay(ex)
%
% Initializes the display for most psychtoolbox experiments 
%
% Adapted from bnaecker@stanford.edu 24 Jan 2013 

%% make sure PTB is working
AssertOpenGL;

%% some default arguments
stereoMode = 0;
screenSize = [];
bgColor = 127.5 .* ones(1, 3);

%% get the display name

% first find the computer name
[f fullCName] = system('scutil --get ComputerName');
% only check on first portion
cname = fullCName(1:10);

% figure out which system we're on
if ~f
	if strncmp(cname, 'Lane', 8) 
		if max(Screen('Screens')) > 1
			dType = 'dell';
		else
			% check if using HPZR30w
			[~, checkStr] = system('system_profiler SPDisplaysDataType');
			if ~isempty(strfind(checkStr, 'ZR30w'));
				dType = 'ZR30w';

				% tell Screen to remap CRTC output
				Screen('Preference', 'ScreenToHead', 0, 1, 1);
			else
				% using the built-in display type
				dType = 'mpb15';
			end
		end

	elseif strncmp(cname, 'Baccus', 6) 
		dType = 'baccusmac';
	elseif strncmp(cname, 'baccuspc', 8);
		dType = 'baccuspc';
	else
		dType = 'unknown';
	end
end

% save computer name to experimental structure
ex.pa.computerName = fullCName;

%% get the screen
ex.ds.screenNum = max(Screen('Screens'));

%% initialize PTB's OpenGL pipeline
InitializeMatlabOpenGL;
Screen('Preference', 'VisualDebugLevel', 3);

%% open a double-buffered window
%[ex.ds.winPtr ex.ds.winRect] = ...
%	Screen('OpenWindow', ex.ds.screenNum, bgColor, screenSize);

%% try out psycimaging stuff
% prepare setup of imaging pipeline
PsychImaging('PrepareConfiguration');

% add support for fast offscreen windows
PsychImaging('AddTask', 'General', 'UseFastOffscreenWindows');

% open the window
[ex.ds.winPtr, ex.ds.winRect] = PsychImaging('OpenWindow', ...
	ex.ds.screenNum, bgColor, screenSize);

%% get information about the screen
ex.ds.ifi = Screen('GetFlipInterval', ex.ds.winPtr);
ex.ds.frate = round(1 / ex.ds.ifi);
ex.ds.winCtr = ex.ds.winRect(3:4) ./ 2;
ex.ds.info = Screen('GetWindowInfo', ex.ds.winPtr);

%% colors
ex.ds.white = WhiteIndex(ex.ds.screenNum);
ex.ds.black = BlackIndex(ex.ds.screenNum);
ex.ds.gray = (ex.ds.black + ex.ds.white) / 2;

%% text information
Screen('TextFont', ex.ds.winPtr, 'Helvetica');
Screen('TextSize', ex.ds.winPtr, 20);
Screen('TextStyle', ex.ds.winPtr, 1);
