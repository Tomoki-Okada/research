function Hawkes_Simulation(mu, A, w)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Haweks過程をfast thning methodでシミュレーションするプログラム
%
% インプット:
% mu = [mu1; mu2] - 2*1のbase intensityベクトル
% A = [a11, a22; a21, a22] - 2*2のintensityジャンプ行列
% w - intenstyの減衰強度
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

options.N = 50; % the number of sequences
options.Nmax = 500; % the maximum number of events per sequence
options.Tmax = 1; % the maximum size of time window
options.tstep = 0.2;% the step length for computing sup intensity
options.M = 50; % the number of steps
options.GenerationNum = 5; % the number of generations
D = 2; % the dimension of Hawkes processes
nTest = 5;
nSeg = 5;
nNum = options.N/nSeg;

disp('Fast simulation of Hawkes processes with exponential kernel')
para.mu = mu;
para.A = A;
para.A = reshape(para.A, [D, 1, D]);
para.w = w;
Seqs = SimulationFast_Thinning_ExpHP(para, options);
data = [];
for i=1:options.N
    Time = Seqs(i).Time.';
    Mark = Seqs(i).Mark.';
    data = [data; [i, i]; [Time, Mark]];
end   
csvwrite("simulation.csv",data);
