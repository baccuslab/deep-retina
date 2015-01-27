function ex = setupExptKB(ex)
%
% FUNCTION ex = setupExptKB(ex)
%
% Sets up the keyboard for most psychtoolbox experiments. 
%
% (c) bnaecker@stanford.edu 24 Jan 2013 

ListenChar(2); % Stop making keypresses show up in matlab

KbName('UnifyKeyNames');
k.escKey = KbName('ESCAPE');
k.oneKey = KbName('1!');
k.twoKey = KbName('2@');
k.threeKey = KbName('3#');
k.fourKey = KbName('4$');
k.fiveKey = KbName('5%');
k.qKey = KbName('q');
k.wKey = KbName('w');
k.eKey = KbName('e');
k.rKey = KbName('r');
k.spaceKey = KbName('space');
k.pKey = KbName('p');
k.oKey = KbName('o');
k.iKey = KbName('i');
k.kKey = KbName('k');
k.lKey = KbName('l');
k.zKey = KbName('z');
k.xKey = KbName('x');
k.cKey = KbName('c');
k.aKey = KbName('a');
k.sKey = KbName('s');
k.dKey = KbName('d');
k.fKey = KbName('f');
k.nKey = KbName('n');
k.bKey = KbName('b');
k.yKey = KbName('y');
k.jKey = KbName('j');
k.tKey = KbName('t');
k.upArrow = KbName('uparrow');
k.downArrow = KbName('downarrow');
k.leftArrow = KbName('leftarrow');
k.rightArrow = KbName('rightarrow');

returnKey = KbName('return');
k.returnKey = returnKey(1);

% shortcut to some general response keys, e.g., for choices
k.left = k.fKey;
k.right = k.jKey;

if ~IsWin
    % Determine the device number(s)
    devices = getDevices;
    
    % This code is weird
    if ~isempty(devices.keyInputInternal) % If there is an internal device
        k.devInt = devices.keyInputInternal(1);
        disp(sprintf('Internal Keyboard: %i',  k.devInt));
    end
    if length(devices.keyInputExternal)==1 % If there is an external Keyboard
        k.devExt = devices.keyInputExternal(1);
        disp(sprintf('External Keyboard: %i',  k.devExt));
        
    elseif length(devices.keyInputExternal)==2 % If there are multiple external keyboards
        k.devExt = devices.keyInputExternal(1);
        k.devExt2 = devices.keyInputExternal(2);
        disp('External keyboard 1');
        disp(k.devExt)
        disp('External keyboard 2');
        disp(k.devExt2)
    end
end

% Initialize KbCheck
[k.keyIsDown, k.secs, k.keyCode] = KbCheck;

% order the structure
k = orderfields(k);

%% Output to ex.kb
for r = 1:length(ex.pa)
    ex.kb(r) = k;
end
