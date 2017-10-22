function N = aggregator(data)

N_1 = zeros(10,1);
N_2 = zeros(10,1);
for i=1:length(data)
    for t=1:10
        if (t-1)*0.1 <= data(i,1) && data(i,1) < t*0.1
            if data(i,2) == 1
                N_1(t) = N_1(t) + 1;
            elseif data(i,2) == 2
                N_2(t) = N_2(t) + 1;
            end
        end
    end
end
N = [transpose(1:10), N_1, N_2, N_1-N_2];
                
         
                
            