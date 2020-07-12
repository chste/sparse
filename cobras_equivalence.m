clear all;
warning off

% spatial frequencies -1 <= fs < 1
fs = [0.5, 0.1];
% number of sources
NSrc = length(fs);
% source covariance matrix
rho = 0; 
Rss = [1 rho ; rho' 1];
% number of snapshots
NSnp = 10;
% signal to noise ratio
SNR = 10;
% function for steering vectors in terms of spatial frequency -1 <= fs <= 1
aFs = @(g,fs) exp(-1i*pi*(real(g(:))*fs(:)'));

% number of subarrays
NArr = 4;
% number of sensors per subarray
NSen0 = 3;
% total number of sensor in array
NSen = NArr*NSen0;

% subarray geometies
for kArr = 1:NArr
    GArr{kArr} = 1:NSen0;
end

% position of subarrays
r = (0:NArr-1)*NSen0;

% overall array geometry
GAll = [];
for k=1:NArr    
    % compute global positions
    GAll = [GAll; r(k)+GArr{k}(:)];   
end

IM = eye(NSen);

% grid for DOA estimation
NGrd = 100;
fsG = (0:NGrd-1)*2/(NGrd)-1;

% compute rearranged steering matrix
C = [];
for kGrd = 1:NGrd
    Cn = [];
    for kArr = 1:NArr
        Cn = blkdiag(Cn, aFs(GArr{kArr}, fsG(kGrd)));
    end
    C = [C, Cn];
end

%% signal generation

% noise power
sigma_N_sqr = 10.^(-SNR/10);

% source signal power
sigma_S_sqr = ones(NSrc,1);

% matrix of source signal snapshots
S = sqrtm(Rss/2)*(randn(NSrc,NSnp) + 1i*randn(NSrc,NSnp));

% compute overall signals
Y0 = aFs(GAll,fs)*S;
N0 = sqrt(sigma_N_sqr/2)*(randn(NSen,NSnp,1) + 1i*randn(NSen,NSnp,1));
Y = Y0 + N0;
R = Y*Y'/NSnp;

% regularization parameter
lambda = sqrt(sigma_N_sqr*NSen0*log(NSen));

%% nuc-norm problem -- grid based

t0 = tic;
cvx_begin quiet
        
    variable X_sdp(NArr+NSnp,NArr+NSnp,NGrd) hermitian semidefinite
    
    l1 = 0;
    YRes = 0;
    
    for kGrd = 1:NGrd       
        
        Qn = X_sdp(1:NArr,NArr+1:end,kGrd);
        U = X_sdp(1:NArr,1:NArr,kGrd);
        V = X_sdp(NArr+1:end,NArr+1:end,kGrd);
        
        l1 = l1 + trace(U) + trace(V);
        YRes = YRes + C(:,(kGrd-1)*NArr+1:kGrd*NArr)*Qn;
        
    end
    
    minimize 1/2*square_pos(norm(YRes - Y,'fro')) + 1/2*lambda*sqrt(NSnp)*l1  
    
cvx_end
tNuc = toc(t0);

cNuc = cvx_optval;

QNuc = [];
sv_Qn_Nuc = [];
for kGrd = 1:NGrd       

    Qn = X_sdp(1:NArr,NArr+1:end,kGrd);
    QNuc = [QNuc; Qn];
    sv_Qn_Nuc = [sv_Qn_Nuc, svd(Qn)];
end

%% COBRAS problem -- grid-based

if 1 % NSnp >= NSen
    
    t0 = tic;
    cvx_begin sdp quiet
        
        variable S_est(NArr,NArr,NGrd) hermitian semidefinite;
        variable U_smr2(NSen,NSen) hermitian;
        
        Q0_grid = 0;
        l1 = 0;
        for kGrd = 1:NGrd
            idxCo = (kGrd-1)*NArr+1:kGrd*NArr;
            Q0_grid = Q0_grid + C(:,idxCo)*S_est(:,:,kGrd)*C(:,idxCo)';
            l1 = l1 + trace(S_est(:,:,kGrd));
        end
        
        minimize real(trace(U_smr2*R)) + l1;
        subject to        
            [U_smr2, IM; IM, Q0_grid+lambda*IM] >= 0;
        
    cvx_end
    tCo = toc(t0);
        
else
    
    t0 = tic;
    cvx_begin sdp quiet
        
        variable S_est(NArr,NArr,NGrd) hermitian semidefinite;
        variable U_smr2(NSnp,NSnp) hermitian;
        
        Q0_grid = 0;
        l1 = 0;
        for kGrd = 1:NGrd
            idxCo = (kGrd-1)*NArr+1:kGrd*NArr;
            Q0_grid = Q0_grid + C(:,idxCo)*S_est(:,:,kGrd)*C(:,idxCo)';
            l1 = l1 + trace(S_est(:,:,kGrd));
        end
        
        minimize real(trace(U_smr2))/NSnp + l1;
        subject to        
            [U_smr2, Y'; Y, Q0_grid+lambda*IM] >= 0;
        
    cvx_end
    tCo = toc(t0);
        
end
cCo = cvx_optval;

SCo = [];
sv_Qn_Co = [];
for kGrd = 1:NGrd       
    S2n = S_est(:,:,kGrd);
    SCo = blkdiag(SCo, S2n);
    sv_Qn_Co = [sv_Qn_Co, svd(S2n)];
end

QCo = SCo*C'*inv(C*SCo*C'+lambda*IM)*Y;

% [pks,idx] = findpeaks(sv_Qn_mn(1,:),'SORTSTR','descend');
% fspG = fsG(idx(1:NSrc));

%% plot results

errQ = norm(QNuc-QCo,'fro')

fprintf('Computation time for ...\n');
fprintf(' Nuclear Norm: %.2fs\n', tNuc);
fprintf(' COBRAS: %.2fs\n\n', tCo);
fprintf('Difference in solutions: %e\n', errQ);

figure(1); hold off;
stem(fsG, sv_Qn_Nuc(1,:)/sqrt(NSnp),'x-r'); hold on;
stem(fsG, sv_Qn_Co(1,:),'o-b'); hold on;
stem(fs, norms(S,2,2)/sqrt(NSnp),'.g');
xlabel('Spatial frequency grid');
ylabel('Block Singular Values');
legend('Mixed-Norm','COBRAS','True')

figure(2); clf;
subplot(1,2,1); hold off;
imagesc(abs(QNuc)); colorbar;
title('Nuclear Norm Solution');
xlabel('Snapshot');
ylabel('Spatial frequency grid');
subplot(1,2,2); hold off;
imagesc(abs(QCo)); colorbar;
title('COBRAS Solution');
xlabel('Snapshot');
ylabel('Spatial frequency grid');