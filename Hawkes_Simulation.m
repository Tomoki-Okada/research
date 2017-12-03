function data = Hawkes_Simulation(mu, A, w)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Haweks�ߒ���fast thning method�ŃV�~�����[�V��������v���O����
%
% �C���v�b�g:
% mu = [mu1; mu2] - 2*1��base intensity�x�N�g��
% A = [a11, a22; a21, a22] - 2*2��intensity�W�����v�s��
% w - intensty�̌������x
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

for i = 1:10
    Time = Seqs(i).Time.';
    Mark = Seqs(i).Mark.';
    csvwrite(strcat('data1/sim', int2str(i), '.csv'),[Time,Mark]);
end

%data = []; 
%for i=1:options.N
%    Time = Seqs(i).Time.';
%    Mark = Seqs(i).Mark.';
%    sim = [Time, Mark];
%    count = aggregator(sim);
%    temp = [(zeros(10,1) + i), count];
%    data = [data; temp];
%end
%csvwrite("simulation.csv",data)
