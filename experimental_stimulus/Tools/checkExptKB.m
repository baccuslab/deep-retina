function ex = checkExptKB(ex)
%
% FUNCTION ex = checkExptKB(ex)
%
% The function checkExptKB polls the keyboard for most psychtoolbox experiments.
%
% (c) bnaecker@stanford.edu 24 Jan 2013 

%% check the keyboard
[ex.kb.keyIsDown, ex.kb.secs, ex.kb.keyCode] = KbCheck(-1);

%% do something
% if ex.kb.keyIsDown
% end
