%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Matlab implementaiton of COBRAS and SI-SPARROW methods presented in the 
% papers:
%
% [1] C. Steffens and M. Pesavento, "Block- and Rank-Sparse Recovery for 
% Direction Finding in Partly Calibrated Arrays," IEEE Transactions on 
% Signal Processing, vol. 66 , no. 2, pp. 384-399, Jan. 2018 
%
% [2] C. Steffens, W. Suleiman, A. Sorg, and M. Pesavento, "Gridless 
% Compressed Sensing Under Shift-Invariant Sampling," Proceedings of the 
% 42nd IEEE International Conference on Acoustics, Speech and Signal 
% Processing (ICASSP), New Orleans, USA, Mar. 2017 
%
% The script uses the CVX optimization toolboxes available at 
% http://cvxr.com/cvx/
%
% The given simulation considers a partly calibrated array composed of 
% identical uniform linear subarrays. However, other array types are
% also supported, as discussed in the above papers.
%
% Author: Christian Steffens
% Date: 24.06.2020
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off
clear;
cvx_precision best

%% signal parameters

% spatial frequencies -1 <= fs < 1
fs = [0.3, -0.1];
% number of source signals
NSrc = length(fs);
% source covariance matrix
rho = 0; % exp(2i*pi/3);
Rss = [1 rho ; rho' 1];
% number of snapshots
NSnp = 10;
% signal to noise ratio in decibel
SNR = 20;
% noise power
sigma_N_sqr = 10.^(-SNR/10);

%% array parameters

% number of subarrays
NArr = 3;
% number of sensors per subarray
NSen0 = 4;
% overall number of sensors in array
NSen = NArr*NSen0;
% position of subarrays
r = [0, 5, 13, 23];

% function for steering vectors in terms of spatial frequency -1 <= fs <= 1
aFs = @(g,fs) exp(-1i*pi*(real(g(:))*fs(:)'));

% subarray geometries, i.e. sensor positions within subarrays
for kArr = 1:NArr
    GArr{kArr} = 0:NSen0-1;
end

% array geometry
GAll = [];
for k=1:NArr
    % number of sensors
    NSenk(k) = length(GArr{k});        
    % global sensor positions
    GAll = [GAll; r(k)+GArr{k}(:)];   
end

% grid for DOA estimation
NGrd = 300;
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


%% identity and selection matrices
IM = eye(NSen);
IM0 = eye(NSen0);
IA = eye(NArr);

% selection matrix for identical subarrays in gridless COBRAS dual problem
J0 = 0;
for m = 1:NSen0
    for n = 1:NArr
        U = zeros(NSen0,NArr);
        U(m,n) = 1;
        J0 = J0+kron(U,U');
    end
end

% selection matrices for SI-SPARROW
for kArr = 1:NArr
    JArr(:,:,kArr) = kron(IA(:,kArr),IM0);
end
% multiple shift-invariances
for kSen1 = 1:NSen0-1
    for kSen2 = 1:NSen0-kSen1+1
        JSen{kSen1}(:,:,kSen2) = kron(IA,IM0(:,kSen2:kSen2+kSen1-1));
    end
end

Ik = kron(ones(NSen0,1),eye(NArr));


%% signal generation

% source signal power
sigma_S_sqr = ones(NSrc,1);

% matrix of source signal snapshots
S = sqrtm(Rss/2)*(randn(NSrc,NSnp) + 1i*randn(NSrc,NSnp));

% compute overall signals
Y0 = aFs(GAll,fs)*S;
N0 = sqrt(sigma_N_sqr/2)*(randn(sum(NSenk),NSnp,1) + 1i*randn(sum(NSenk),NSnp,1));
Y = Y0 + N0;
R = Y*Y'/NSnp;

Q = aFs(GAll,fs)*diag(norms(S,2,2))*aFs(GAll,fs)';

% regularization parameter
lambda = sqrt(sigma_N_sqr*max(NSenk))*(sqrt(sum(NSenk)/NSnp)+1);

%% nuc-norm problem -- grid based, see [1]
% becomes inefficient for large number of snapshots or grid points

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

% rearrange matrices for signal matrix reconstuction
QNuc = [];
sv_QNuc = [];
for kGrd = 1:NGrd       
    QnNuc = X_sdp(1:NArr,NArr+1:end,kGrd);
    QNuc = [QNuc; QnNuc];
    sv_QNuc = [sv_QNuc, svd(QnNuc)];
end


%% grid-based COBRAS, see [1]
% computational complexity is not affected by large number of snapshots

t0 = tic;

% formulation based on signal covariance matrix R
if NSnp >= NSen 
    
    
    cvx_begin sdp quiet
        
        variable S_est(NArr,NArr,NGrd) hermitian semidefinite;
        variable U(NSen,NSen) hermitian;
        
        Q0_grid = 0;
        l1 = 0;
        for kGrd = 1:NGrd
            idxCo = (kGrd-1)*NArr+1:kGrd*NArr;
            Q0_grid = Q0_grid + C(:,idxCo)*S_est(:,:,kGrd)*C(:,idxCo)';
            l1 = l1 + trace(S_est(:,:,kGrd));
        end
        
        minimize real(trace(U*R)) + l1;
        subject to        
            [U, IM; IM, Q0_grid+lambda*IM] >= 0;
        
    cvx_end
        
% formulation based on signal matrix Y
else
    
    cvx_begin sdp quiet
        
        variable S_est(NArr,NArr,NGrd) hermitian semidefinite;
        variable U(NSnp,NSnp) hermitian;
        
        Q0_grid = 0;
        l1 = 0;
        for kGrd = 1:NGrd
            idxCo = (kGrd-1)*NArr+1:kGrd*NArr;
            Q0_grid = Q0_grid + C(:,idxCo)*S_est(:,:,kGrd)*C(:,idxCo)';
            l1 = l1 + trace(S_est(:,:,kGrd));
        end
        
        minimize real(trace(U))/NSnp + l1;
        subject to        
            [U, Y'; Y, Q0_grid+lambda*IM] >= 0;
        
    cvx_end
    
end

tGbCobras = toc(t0);
cGbCobras = cvx_optval;

% signal reconstruction
SCobras = [];
sv_QCobras = [];
for kGrd = 1:NGrd
    S2n = S_est(:,:,kGrd);
    SCobras = blkdiag(SCobras, S2n);
    sv_QCobras = [sv_QCobras, svd(S2n)];
end

SNuc = SCobras*C'*inv(C*SCobras*C'+lambda*IM)*Y;

%% gridless COBRAS - primal problem with block-toeplitz structure, see [1]

t0 = tic;
if NSnp >= NSen
        
    cvx_begin sdp quiet

        variable Q0t(NSen,NSen) hermitian;
        variable Ut(sum(NSen),sum(NSen)) hermitian;
        variable Dtoept(NSen0,NArr) complex;
        variable Otoept(2*NSen0-1,(NArr^2-NArr)/2) complex;
        dual variable P1t;
        dual variable P2t;

        minimize real(trace(Ut*R)) + real(trace(Q0t))/NSen0
        subject to
            P1t : [Ut, IM; IM, Q0t + lambda*IM] >= 0;
            P2t : Q0t >= 0;
            
            k0 = 1;
            for kArr1 = 1:NArr
                idx1 = (kArr1-1)*NSen0+1:kArr1*NSen0;
                Q0t(idx1,idx1) == toeplitz(Dtoept(:,kArr1));
                Q0t(idx1,idx1) >= 0;
                for kArr2 = 1:kArr1-1
                    idx2 = (kArr2-1)*NSen0+1:kArr2*NSen0;
                    Q0t(idx1,idx2) == toeplitz(Otoept(NSen0:-1:1,k0),Otoept(NSen0:end,k0));
                    k0 = k0+1;
                end
            end
    cvx_end    
        
else
    
    cvx_begin sdp quiet

        variable Q0t(NSen,NSen) hermitian;
        variable Ut(NSnp,NSnp) hermitian;
        variable Dtoept(NSen0,NArr) complex;
        variable Otoept(2*NSen0-1,(NArr^2-NArr)/2) complex;        
        dual variable P1t;
        dual variable P2t;

        minimize real(trace(Ut))/NSnp + real(trace(Q0t))/NSen0
        subject to
            P1t : [Ut, Y'; Y, Q0t + lambda*IM] >= 0;
            P2t : Q0t >= 0;
            k0 = 1;
            for kArr1 = 1:NArr
                idx1 = (kArr1-1)*NSen0+1:kArr1*NSen0;
                Q0t(idx1,idx1) == toeplitz(Dtoept(:,kArr1));
                Q0t(idx1,idx1) >= 0;
                for kArr2 = 1:kArr1-1
                    idx2 = (kArr2-1)*NSen0+1:kArr2*NSen0;
                    Q0t(idx1,idx2) == toeplitz(Otoept(NSen0:-1:1,k0),Otoept(NSen0:end,k0));
                    k0 = k0+1;
                end
            end

    cvx_end
    
end

tGlCobrasPr = toc(t0);
cGlCobrasPr = cvx_optval;

% get signal frequencies by ESPRIT method
gevp1 = eig(JSen{NSen0-1}(:,:,1)'*Q0t*JSen{NSen0-1}(:,:,1), JSen{NSen0-1}(:,:,2)'*Q0t*JSen{NSen0-1}(:,:,1));
[~,idx] = sort(abs(abs(gevp1)-1),'ascend');
gevp1 = gevp1(idx(1:NSrc));
fsGlCobrasPr = sort(angle(gevp1)/pi,'ascend');

%% gridless COBRAS, dual problem, see [1]

t0 = tic;
if NSnp >= NSen
        
    cvx_begin sdp quiet
        
        variable Z1(NSen,NSen) complex;
        variable Z0(NSen,NSen) hermitian semidefinite;
        variable H(NSen,NSen) hermitian;
        dual variable D1;
        dual variable D2;
        
        % block diagonal constraints on gram matrix
        for kSen=1:NSen0
            Jk = kron(diag(ones(NSen0-kSen+1,1),kSen-1),ones(NArr));
            Ck(:,:,kSen) = Ik'*(H.*Jk)*Ik;
        end
        
        squeeze(Ck(:,:,1)) == eye(NArr);
        for kSen = 2:NSen0
            squeeze(Ck(:,:,kSen)) == zeros(NArr);
        end
        
        % semidefinite constraints
        D1 : [R, Z1'; Z1, Z0] >= 0;
        D2 : H - J0*Z0*J0' >= 0;    

        % obective function
        maximize -2*real(trace(Z1)) - lambda*real(trace(Z0));                    

    cvx_end    

else
    
    cvx_begin sdp quiet

        variable Z1(NSen,NSnp) complex;
        variable Z0(NSen,NSen) hermitian semidefinite;
        variable H(sum(NSen),sum(NSen)) hermitian;
        dual variable D1;
        dual variable D2;

        % block diagonal constraints on gram matrix
        for kSen=1:NSen0
            Jk = kron(diag(ones(NSen0-kSen+1,1),kSen-1),ones(NArr));
            Ck(:,:,kSen) = Ik'*(H.*Jk)*Ik;
        end

        squeeze(Ck(:,:,1)) == eye(NArr);
        for kSen = 2:NSen0
            squeeze(Ck(:,:,kSen)) == zeros(NArr);
        end

        % semidefinite constraints
        D1 : [eye(NSnp)/NSnp, Z1'; Z1, Z0] >= 0;
        D2 : H - J0*Z0*J0' >= 0;    

        % obective function
        maximize -2*real(trace(Z1*Y')) - lambda*real(trace(Z0));                    

    cvx_end

end

tGlCobrasDu = toc(t0);
cGlCobrasDu = cvx_optval;

% compute matrix polynomial coefficients for frequency estimation
Ck = [];
for kSen=-NSen0+1:NSen0-1
    Jk = kron(diag(ones(NSen0-abs(kSen),1),kSen),ones(NArr));
    Ck(:,:,kSen+NSen0) = -Ik'*((J0*Z0*J0').*Jk)*Ik;
end
Ck(:,:,NSen0) = eye(NArr)+squeeze(Ck(:,:,NSen0));

% compute block companion matrix pair
V1 = [zeros((2*NSen0-3)*NArr,NArr), eye((2*NSen0-3)*NArr); 
      -reshape(Ck(:,:,1:end-1),NArr,2*(NSen0-1)*NArr,[])];
V2 = [eye((2*NSen0-3)*NArr), zeros((2*NSen0-3)*NArr,NArr); 
      zeros(NArr,(2*NSen0-3)*NArr), Ck(:,:,end)];

gevdMP = eig(V2,V1);

% isolate roots on unit circle
th1 = 1e-2;
gevdMP = gevdMP(abs(gevdMP) >= 1-th1 & abs(gevdMP) <= 1+th1);

th2 = 1e-2;
[~,ia,ic] = unique(round(angle(gevdMP)/th2)*th2);
gevdMP = gevdMP(ia);

NSrcEst = length(gevdMP);
fsGlCobrasDu = sort(angle(gevdMP)/pi,'ascend');

%% SI-SPARROW

% Gridless SPARROW for arrays with shift-invariances (SI).
% As discussed in the paper [2], multiple kinds of shift-invariances can be 
% exploited here. For a fair comparison with GL-COBRAS, only the identical 
% subarray structure is exploited, as represented by the selection
% matrices JSen

t0 = tic;
if NSnp >= NSen
    
    cvx_begin sdp quiet

        variable Q0(NSen,NSen) hermitian;
        variable U(sum(NSen),sum(NSen)) hermitian;
        dual variable P1;
        dual variable P2;

        minimize real(trace(U*R)) + real(trace(Q0))/NSen0
        subject to
            P1 : [U, IM; IM, Q0 + lambda*IM] >= 0;
            P2 : Q0 >= 0;
            for kSen1 = 1:NSen0-1
                for kSen2 = 2:NSen0-kSen1+1
                    JSen{kSen1}(:,:,1)'*Q0*JSen{kSen1}(:,:,1) == JSen{kSen1}(:,:,kSen2)'*Q0*JSen{kSen1}(:,:,kSen2);
                end
            end

    cvx_end
    
else
    
    cvx_begin sdp quiet

        variable Q0(NSen,NSen) hermitian;
        variable U(NSnp,NSnp) hermitian;
        dual variable P1;
        dual variable P2;

        minimize real(trace(U))/NSnp + real(trace(Q0))/NSen0
        subject to
            P1 : [U, Y'; Y, Q0 + lambda*IM] >= 0;
            P2 : Q0 >= 0;
%             for kArr = 1:NArr
%                 JArr(:,:,1)'*Q0*JArr(:,:,1) == JArr(:,:,kArr)'*Q0*JArr(:,:,kArr);
%             end
            for kSen1 = 1:NSen0-1
                for kSen2 = 2:NSen0-kSen1+1
                    JSen{kSen1}(:,:,1)'*Q0*JSen{kSen1}(:,:,1) == JSen{kSen1}(:,:,kSen2)'*Q0*JSen{kSen1}(:,:,kSen2);
                end
            end

    cvx_end
        
end

tSiSparrow = toc(t0);
cSiSparrow = cvx_optval;

% estimation of signal frequencies by ESPRIT method
gevp1 = eig(JSen{NSen0-1}(:,:,1)'*Q0*JSen{NSen0-1}(:,:,1), JSen{NSen0-1}(:,:,2)'*Q0*JSen{NSen0-1}(:,:,1));
[~,idx] = sort(abs(abs(gevp1)-1),'ascend');
gevp1 = gevp1(idx(1:NSrc));
fsSiSparrow = sort(angle(gevp1)/pi,'ascend');


%% plot results

fprintf('\nFrequency estimates:\n');
fprintf('Ground Truth           | %s\n', sprintf('%.2f, ', sort(fs, 'ascend')));
fprintf('Gridless COBRAS Primal | %s\n', sprintf('%.2f, ', sort(fsGlCobrasPr)));
fprintf('Gridless COBRAS Dual   | %s\n', sprintf('%.2f, ', sort(fsGlCobrasDu)));
fprintf('SI-SPARROW             | %s\n\n', sprintf('%.2f, ', sort(fsSiSparrow)));

fprintf('Solver metrics:\n');
fprintf('                       | Comp. Time | Opt. Value \n');
fprintf('------------------------------------------------ \n');
fprintf('Block-Nuclear Norm     | %9.2fs |  %9.2f \n', tNuc, cNuc/(lambda*NSnp/2));
fprintf('Grid-based COBRAS      | %9.2fs |  %9.2f \n', tGbCobras, cGbCobras);
fprintf('Gridless COBRAS Primal | %9.2fs |  %9.2f \n', tGlCobrasPr, cGlCobrasPr);
fprintf('Gridless COBRAS Dual   | %9.2fs |  %9.2f \n', tGlCobrasDu, cGlCobrasDu);
fprintf('SI-SPARROW             | %9.2fs |  %9.2f \n', tSiSparrow, cSiSparrow);

figure(1); clf;
subplot(2,1,1); hold off;
cs = 'rgbymc';
for kArr=1:NArr
    plot(r(kArr)+GArr{kArr}(:), zeros(NSen0,1), ['o', cs(kArr)], 'MarkerFaceColor',cs(kArr)); 
    ls{kArr} = sprintf('Subarray %d', kArr);
    subarrayCenter = r(kArr)+(min(GArr{kArr})+max(GArr{kArr}))/2;
    text(subarrayCenter, 2, sprintf('Subarray %d', kArr), 'HorizontalAlignment','center');
    
    hold on;
end
arrayCenter = (min(GAll)+max(GAll))/2;
arrayLenght = max(GAll)-min(GAll);
for kSrc = 1:NSrc
    srcAngle = acosd(fs(kSrc));
    plot([arrayCenter, arrayCenter + arrayLenght*cosd(srcAngle)], [3, 3 + arrayLenght*sind(srcAngle)], 'k') 
    text(arrayCenter + arrayLenght*cosd(srcAngle), 4 + arrayLenght*sind(srcAngle), sprintf('source %d', kSrc), 'HorizontalAlignment','center');
end
title('Simulation Setup');
xlabel('x-position in half wavelength');
ylabel('y-position in half wavelength');

subplot(2,1,2); hold off;
stem(fsG, sv_QNuc(1,:), 'ob'); hold on;
stem(fsG, sv_QCobras(1,:)*sqrt(NSnp), 'xr'); 
xlabel('spatial frequency fs');
ylabel('singular values');
legend('Block-Nuclear Norm','Grid-Based COBRAS');
title('Spatial Spectra for Grid-Based Methods');

