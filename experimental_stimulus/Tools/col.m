% returns col vector
% Niru Maheswaranathan
% Thu Jul  5 11:50:11 2012

function cv = col(v)

    s = size(v);
    if s(1) == 1
        cv = v';
    elseif s(2) == 1
        cv = v;
    else
        error('The input must be a vector.');
    end

end
