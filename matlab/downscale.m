function [ Id, Dd, Kd ] = downscale( I, D, K, level )
    if(level <= 1)
        Id = I;
        Dd = D;
        Kd = K;
        return;
    end

    % this is because we interpolate in such a way, that 
    % the image is discretized at the exact pixel-values (e.g. 3,7), and
    % not at the center of each pixel (e.g. 3.5, 7.5).
    Kd = [K(1,1)/2 0 (K(1,3)+0.5)/2-0.5;
          0 K(2,2)/2 (K(2,3)+0.5)/2-0.5;
          0 0 1];

    Id = (I(0+(1:2:end), 0+(1:2:end)) + ...
          I(1+(1:2:end), 0+(1:2:end)) + ...
          I(0+(1:2:end), 1+(1:2:end)) + ...
          I(1+(1:2:end), 1+(1:2:end)))*0.25;

    DdCountValid = (sign(D(0+(1:2:end), 0+(1:2:end))) + ...
                    sign(D(1+(1:2:end), 0+(1:2:end))) + ...
                    sign(D(0+(1:2:end), 1+(1:2:end))) + ...
                    sign(D(1+(1:2:end), 1+(1:2:end))));

    Dd = (D(0+(1:2:end), 0+(1:2:end)) + ...
            D(1+(1:2:end), 0+(1:2:end)) + ...
            D(0+(1:2:end), 1+(1:2:end)) + ...
            D(1+(1:2:end), 1+(1:2:end))) ./ DdCountValid;
    Dd(isnan(Dd)) = 0;

    [Id, Dd, Kd] = downscale( Id, Dd, Kd, level -1 );
end
