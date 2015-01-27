function devices=getDevices;
% devices=getDevices - gets device numbers for KbCheck;
%
% Usefull for getting a summary of device numbers. 
% These device numbers are needed for an input argument for 
% KbCheck if you have more than one connected (otherwise
% it will pick the first one). 
%
% Also tries to determine whether input is 'internal' or 'external'.
%
% 07/2005 SOD wrote it. 

% get all devices
d = PsychHID('Devices');

% make output struct
devices.keyInputInternal = [];
devices.keyInputExternal = [];

% loop over all devices and try kbCheck
disp(sprintf('[%s]:Getting device information and testing KbCheck:',mfilename));
for n=1:length(d),
    fprintf(1,'   Device %d: %s (%s): ',n,d(n).product,d(n).transport);
    try,
        evalc(sprintf('KbCheck(%d)',n)); % we don't need the error output
        fprintf(1,'KbCheck(%d) OK.\n',n);
        % store and organize 'good' device numbers
        internal = 0;
        if strcmp(lower(d(n).transport),'adb'),
            internal = 1;
        end;
        if length(d(n).product) >= 14,
            if strcmp(lower(d(n).product(1:14)),'apple internal'),
                internal = 1;
            end;
        end;
        
       if internal,
           devices.keyInputInternal = [devices.keyInputInternal n];
       else, % we assume it is an external device
           devices.keyInputExternal = [devices.keyInputExternal n];
       end;
    catch,
        fprintf(1,'KbCheck(%d) failed.\n',n);
    end;
end;
