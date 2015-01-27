% returns row vector
% Niru Maheswaranathan
% Thu Jul  5 11:50:11 2012

function rv = row(v)

    s = size(v);
    if s(1) == 1
        rv = v;
    elseif s(2) == 1
        rv = v';
    else
        error('The input must be a vector.');
    end

end
