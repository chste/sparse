% test equivalence of mixed-norm minimization, atomic norm minimization and SPARROW

clear;
warning off;

%% simulation parameters
% spatial frequencies
fs = [0.3 0.1 -0.5505];
% number of sources
NSrc = length(fs);
% function for steering vectors in terms of spatial frequency -1 <= fs < 1
aFs = @(g,fs) exp(-1i*pi*(g(:)*fs(:)'));
% number of sensors
NSen = 20;
% number of snapshots 
NSnp = 30;
% Signal-to-Noise ratio in dB
SNR = 20;
% uniform array geometry
G = 0:NSen-1;  
% number of grid points
NGrd = 500;

%% signal generation

% steering matrix
fs = sort(fs,'ascend');
A0 = aFs(G, fs);

% grid for DOA estimation and dictionary matrix
fsG = 2*(0:NGrd-1)/NGrd-1;
A = aFs(G, fsG);

% generate signal
sigma_N_sqr = 10.^(-SNR/10);

% matrix of source signal snapshots
X0 = sqrt(1/2)*(randn(NSrc,NSnp) + 1i*randn(NSrc,NSnp));

% compute overall signals
Y0 = aFs(G,fs)*X0;
N0 = sqrt(sigma_N_sqr/2)*(randn(NSen,NSnp) + 1i*randn(NSen,NSnp));            
Y = Y0 + N0;
R = Y*Y'/NSnp;

% compute regularization parameter
lambda = sqrt(sigma_N_sqr*NSen*log(NSen));

%% Mixed-Norm Minimization

t1 = tic;
cvx_begin quiet

    variable X_mn(NGrd,NSnp) complex

    minimize 1/2*square_pos(norm(Y-A*X_mn,'fro')) + lambda*sqrt(NSnp)*sum(norms(X_mn,2,2))

cvx_end
t_mn = toc(t1);
c_mn = cvx_optval;

s_mn = norms(X_mn,2,2)/sqrt(NSnp);

%% SPARROW

IM = eye(NSen);

t1 = tic;
cvx_begin quiet
    
    variable s_sp(NGrd,1) nonnegative;
    variable Q_sp(NSen,NSen) hermitian semidefinite;
    
    minimize real(trace(Q_sp*R)) + sum(s_sp)
    subject to        
        [Q_sp, IM; IM, A*diag(s_sp)*A'+lambda*eye(NSen)] == hermitian_semidefinite(2*NSen);
    
cvx_end
t_sp = toc(t1);
c_sp = lambda*NSnp/2*cvx_optval;

X_sp = diag(s_sp)*A'*inv(A*diag(s_sp)*A'+lambda*eye(NSen))*Y;
errX = norm(X_mn-X_sp,'fro')/norm(X_mn,'fro');

fprintf('Optimal value --    Mixed-Norm: %8.5f,   SPARROW: %8.5f\n', c_mn, c_sp);
fprintf('Computation Time -- Mixed-Norm: %8.5f,   SPARROW: %8.5f\n', t_mn, t_sp);
fprintf('Relative difference in minimizers  %.2e\n\n', errX);

%% Gridless SPARROW

t1 = tic;
cvx_begin quiet
    
    variable T_glsp(NSen,NSen) hermitian semidefinite toeplitz;
    variable Q_glsp(NSen,NSen) hermitian semidefinite;
    
    minimize real(trace(Q_glsp*R)) + trace(T_glsp)/NSen
    subject to        
        [Q_glsp, IM; IM, T_glsp+lambda*eye(NSen)] == hermitian_semidefinite(2*NSen);
    
cvx_end
t_glsp = toc(t1);
c_glsp = lambda*NSnp/2*cvx_optval;


%% Atomic norm minimization

t1 = tic;
cvx_begin quiet
        
    variable T_anm(NSen,NSen) hermitian toeplitz;
    variable Q_anm(NSnp,NSnp) hermitian;
    variable Y_anm(NSen,NSnp) complex;
    
    minimize 1/2*square_pos(norm(Y-Y_anm,'fro')) + lambda*sqrt(NSnp)/2*real(trace(Q_anm) + trace(T_anm)/NSen)
    subject to        
        [Q_anm, Y_anm'; Y_anm, T_anm] == hermitian_semidefinite(NSen+NSnp);
    
cvx_end
t_anm = toc(t1);
c_anm = cvx_optval;

errT = norm(T_glsp-T_anm/sqrt(NSnp),'fro')/norm(T_glsp,'fro');

fprintf('Optimal value --    Atomic Norm: %8.5f,   GL-SPARROW: %8.5f\n', c_anm, c_glsp);
fprintf('Computation Time -- Atomic Norm: %8.5f,   GL-SPARROW: %8.5f\n', t_anm, t_glsp);
fprintf('Relative difference in minimizers  %.2e\n\n', errT);


%% reconstruct frequencies and magnitudes by Vandermonde decomposition of SPARROW solution

NSrc0 = sum(abs(eig(T_glsp))>1e-3);
u = T_glsp(:,1);

% compute frequencies
g = toeplitz([u(2); conj(u(1:NSen-1))], u(2:min(NSrc0+1,NSen))) \ conj(-u(1:NSen));
z0 = roots([1; g]);
z0 = z0 ./ abs(z0);
fs0 = angle(z0)/pi;

% compute amplitudes
A0 = aFs(G, fs0);
s0 = real(A0\u);
[s0, idx] = sort(s0, 'descend');
fs0 = fs0(idx);


h1 = figure(1); clf;
stem(fsG,s_mn,'o-b'); hold on
stem(fsG,s_sp,'x-r');
stem(fs0,s0,'s-k');
stem(fs,norms(X0,2,2)/sqrt(NSnp),'.-g');
xlabel('Spatial frequency fs');
ylabel('Signal norm s');
legend('Mixed-Norm Min.','SPARROW','GL-SPARROW','True');
