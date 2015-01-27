function ex = saveExptStruct(ex)
%
% FUNCTION ex = saveExptStruct(ex)
%
% Save the required data, assuming that there was no error and that the
% esc key was not pressed
%
% (c) bnaecker@stanford.edu 15 Feb 2012

%% Check for errors and the keyboard
if ~isfield(ex, 'me') && ~ex.kb.keyCode(ex.kb.escKey)
    Screen('DrawText', ex.ds.winPtr, 'Saving data structure...', ...
       50, 50); 
    Screen('Flip', ex.ds.winPtr);
    
    % check for the directory
    if ~exist(ex.pa.saveDir, 'dir')
        mkdir(ex.pa.saveDir);
    end
    save(fullfile(ex.pa.saveDir, [ex.pa.date '.mat']), 'ex');
    fprintf('\n\nExperiment ran successfully.\nExperimental structure saved to...\n%s\n\n', ...
        fullfile(ex.pa.saveDir, [ex.pa.date '.mat']));
elseif ex.kb.keyCode(ex.kb.escKey)
	fprintf(['\n\n\nYou quit the experiment early.\nNothing was saved, but the experimental ' ...
		'structure will be in the workspace\n']);
else
    fprintf(['\n\n\nAn error was thrown. \nNothing was saved, but the experimental '...
        'structure will be output to the workspace.']);
    fprintf(['\n\nError message:\n' ex.me.message '\n\n']);
end
