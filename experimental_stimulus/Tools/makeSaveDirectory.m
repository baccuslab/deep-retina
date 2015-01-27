function ex = makeSaveDirectory(ex)
%
% FUNCTION ex = makeSaveDirectory(ex)
%
% The function makeSaveDirectory makes a directory in which to save the data from the
% given experiment name, which must be a field of ex.pa.
%
% (c) bnaecker@stanford.edu  25 Jan 2013

%% make a date folder
dateFolder = datestr(now, 'ddmmyy');

%% make the path based on the computer on which we're running
if strncmp(ex.pa.computerName, 'Ben', 3)
	ex.pa.naturalImgDir = '~/FileCabinet/Stanford/BaccusLab/Images/Tkacik/';
	ex.pa.saveDir = fullfile('~/FileCabinet/Stanford/BaccusLab/Projects/', ...
		ex.pa.experimentName, 'TestData');
elseif strncmp(ex.pa.computerName, 'Baccus', 6)
	ex.pa.naturalImgDir = '/Users/baccuslab/Desktop/stimuli/Ben/Images/';
	ex.pa.saveDir = fullfile('/Users/baccuslab/Desktop/stimuli/Ben/Data', ...
		ex.pa.experimentName);
elseif strncmp(ex.pa.computerName, 'baccuspc', 8);
	ex.pa.naturalImgDir = ['C:\Documents and Settings\Mike Menz\Desktop\Stimuli' ...
		'\Ben\imgs\'];
	ex.pa.saveDir = ['C:\Documents and Settings\Mike Menz\Desktop\Stimuli' ...
		'\Ben\Data\' ex.pa.experimentName];
elseif strncmp(ex.pa.computerName, 'Lane', 4)
    ex.pa.naturalImgDir = '~/Git/retina';
    ex.pa.saveDir = fullfile('~/Git/retina/Lane_Experiment_2013-02-21/', ...
        ex.pa.experimentName, 'TestData');
end

%% check if previous data folders exist from the same day
dirContents = dir(ex.pa.saveDir);
possibleMatches = strncmp(dateFolder, {dirContents.name}, 6);
alphabet = char(97:122);
previousFiles = sum(possibleMatches);
if previousFiles > 0
	ex.pa.saveDir = fullfile(ex.pa.saveDir, [dateFolder alphabet(previousFiles)]);
else
	ex.pa.saveDir = fullfile(ex.pa.saveDir, dateFolder);
end
