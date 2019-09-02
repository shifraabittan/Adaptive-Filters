%% Shifra Abittan
% Prof Fontaine
% ECE416 Adaptive Filters
% PSet1 Linear Algebra
% Question 5. Sensor Array Signal Model

% This function creates an MxN matrix where each entry is a unit variance,
% zero mean complex Gaussian and all entries are independent. The variance
% of n if n = ni + j*nq is additive, var(ni) + var(nq)
%n = @(M,N) 1/sqrt(2)*(randn(M,N) + j*randn(M,N))

% To generate an Mx1 random Gaussian vector with covariance matrix C and mean
% m, start with n, a unit variance white complex Gaussian, C^1.2 is the
% hermitian sq root or cholesky factor
%x = @(C,m,n) (C^0.5)*n + m

%% (A) Function to output steering matrix S and data matrix A

% S, the steering matrix, is an M-by-L matrix with normalized/unit length
% steering vectors as columns. M represents the number of elements in the
% sensor array, and L represents the number of wave sources. A steering 
% vector has elements of the form exp(-j*k(theta)*r_i) where i ranges from 
% 1 to M. k(theta), the wavenumber vector, is 2*pi/lamda * a(theta)
% where a(theta) = sin(theta)*cos(phi)* a_x + sin(theta)*sin(phi)* a_y +
% cos(theta)* a_z, lambda is a constant for the narrowband case, theta is a
% polar angle ranging from 0 to pi/2 and phi is the azimuthal angle ranging
% from 0 to 2*pi.

% See function at bottom of script

%% (B) Correlation Matrix of u[n]
% Theoretical Correlation Matrix: 
% The correlation matrix of a vector is defined as R = E(u[n]u[n]^H) where 
% each element is E(u[n]_i*(u[n]_j)^*).
% u[n] can be written in matricial form = Sb+v, where S is the matrix of
% steering vectors, b is the column vector of signals. Sb can be
% interpreted as a linear combination of the steering vectors (columns of
% S). The correlation R of Sb is S*R_b*S^H because E(S*b*b^H*S^H)=
% S*E(b*b^H)*S^H = S*R_b*S^H. With the addition of noise, the correlation
% R is just the sum of S*R_b*S^H + noise_var*I because the noise and b
% values are uncorrelated (all cross terms drop out).

% Ergodic Correlation Matrix: Because  u[n] is stationary, i.e. the 
% expectation does not depend on time, the correlation matrix can be 
% approximated from AA^H (outer product). 

% See function at bottom of script 

%% (C)

% Parameters
d_over_lambda = 0.5;
N = 100; %number of snapshots

% Two linear arrays aligned across the x-axis and y-axis. Located at
% (md,0,0) and (0,md,0) where m spans from -10 to 10.
m = (-10:10).';
array_locs = zeros(3,42);
array_locs(1,1:21) = m; %sensors along x_axis
array_locs(2,22:42) = m; %sensors along y_axis
array_locs(:,11) = []; %dont double count origin

% Three sources
% The source and noise variances/power are expressed in terms of dB.
% dB = 10 log10(Power/Ref=1) --> 10^(dB/10) = Power
sig_var_1 = 1;
sig_var_2 = 10^(-0.5); % 5dB below source 1 --> 10^(-5/10)
sig_var_3 = 0.1;       % 10dB below source 1 --> 10^(-10/10) = 0.1
noise_var = 0.01;      % 20dB below source 1 --> 10^(-20/10) = 0.01

sig_var = [sig_var_1; sig_var_2; sig_var_3];

% Angles of Arrival
thetas = [10;20;30];
phis = [20;-20;150];

% A matrix for the given parameters
[S,A] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,noise_var,sig_var);

% Theoretical and Approximate Correlation Matrix R of u[n]
[R,R_hat] = cor(A,S,noise_var,sig_var);


%% (C.1) Eigenanalysis of R to determine theoretical singular values of A

% Eigenanalysis

[right_vecs,eigvals,lft_vecs] = eig(R);
figure
stem(abs(diag(eigvals)))
title('Theoretical R Matrix Singular Values - notice 3 dominant values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% Projection Matrices

% Projection Matrix onto Signal Space.
% To do this, take the eigenvectors corresponding to the L(3) largest/most 
% significant eigenvalues and sum their outerproducts = sum from 1 to
% 3(q*q^H) = Q*Q^H
Q = lft_vecs(:,1:3);
P_S = Q*Q'

% Projection Matrix onto Noise Space.
% Project onto noise subspace = sum of the outerproduct of the remaining
% eigenvectors not used for signal subspace. I - projection_matrix =
% orthogonal complement of the space that the projection matrix projects
% onto. Therefore, I - P_S = P_N
P_N = eye(41) - P_S;

% Check that the singular vectors lie in the signal space by computing 
% |P_n*s(theta_i))| and ensuring it is zero
s1 = S(:,1);
proj_noise_s1=sum(abs(P_N*s1)) %Value is 4.68e-15 ~ 0 --> just numerical precision

s2 = S(:,2);
proj_noise_s2=sum(abs(P_N*s2)) %Value is 4.54e-15 ~ 0 --> just numerical precision

s3 = S(:,3);
proj_noise_s3=sum(abs(P_N*s3)) %Value is 5.08e-15 ~ 0 --> just numerical precision


%% (C.2) Noise-free, v[n]=0

[S_0,A_0] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,0,sig_var)
[R0,R0_hat] = cor(A_0,S_0,0,sig_var) % R and R0 are identical except along the diagonal

% Eigenanalysis of R0
[right_vecs_0,eigvals_0,lft_vecs_0] = eig(R0);
figure
stem(abs(diag(eigvals_0)))
title('No Noise R Matrix Singular Values - notice 3 dominant values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% Notice that the significant singular values are the same for the noise
% and no noise R. However, all of the other singular values are zero for
% the model in which no noise is present (R0). For R, the model with white
% noise, the other singular values are small but present due to the noise.


%% (C.3) SVD Analysis on A

A_SVD = svd(A)

% Graph SVDs
% Notice that the fourth singular value is significantly smaller than the
% third because of the good SNR
figure
stem(1:length(A_SVD),A_SVD)
title('Singular Values of A data matrix')
ylabel('Magnitude of Singular Values')
hold on
plot(repmat(3.5,length(A_SVD),1),linspace(0,10,length(A_SVD)),'--')


%% (C.4) Compute SNR from singular values

% Singular value 4 squared should be approximately equivalent to the noise
% power (they are of same magnitude, as expected)
A_S4 = (A_SVD(4)).^2
noise_var

% The sum of the squares of the first 3 significant singular values should
% be approximately equal to the sum of the power of the 3 source signals 
sum_signal_pwr = sum(sig_var) 
sum_3_singularvals = sum((A_SVD(1:3)).^2) %Expect to be approx equal to signal power

% Actual SNR
SNR = sum_signal_pwr./noise_var
% SNR derived from SVD 
SNR_SVD = sum_3_singularvals./A_S4


%% (C.5) Compute ||R-R_hat|| - spectral norm of R_hat_error matrix

% ||R-R_hat|| is the spectral norm of the error matrix (when comparing the
% theoretical R and ergodically approximated R. For a square matrix, the 
% spectral norm is equivalent to the largest singular value or the maximum 
% sqrt(eig(AA^H))
error = R-R_hat;
outprod_error = error*error';
[~,error_eigs,~] = eig(outprod_error);
spec_norm_error = max(max((error_eigs).^0.5))
% The noise power is smaller than the spectral norm of the error matrix by
% one order of magnitude. The fourth eigenvalue of R is approximately
% equivalent to the noise power.

l2_error = norm(error)' %Computing the L2 norm returns the same value as 
% when the eigenvalues are used to compute the spectral norm


%% (C.6) MVDR and MUSIC Spectrum

% Both the MVDR and MUSIC spectrums are functions of the steering vector.
% The spectrum should be evaluated over a grid of (theta,phi) that vary
% in 2 degree increments.

% Theta/polar angle ranges from 0 to pi/2 = 90 degrees
thetas = 0:2:90;
% Phi/azimuthal angle ranges from 0 to 2*pi = 360 degrees
phis = 0:2:360;

[thetas_,phis_] = meshgrid(thetas,phis);

% Compute P_N for MUSIC spectrum using SVD of A
[A_U,A_SV,A_V] = svds(A,3);
% P_N = I - sum 1,2,3(qiqi^H), qi = eigenvec corresponding with ith
% eigenvalue
A_PS = A_U*A_U';
A_PN = eye(41) - A_PS;

MVDR=zeros(46,181);
MUSIC=zeros(46,181);
for t=1:46
    for p=1:181
        S = steeringvec(thetas(t),phis(p)).';
        MVDR(t,p) = 1./(norm(R^-0.5*S).^2); %same as 1./(S'*inv(R_hat)*S)
        MUSIC(t,p) = 1./(norm(A_PN*S).^2); %same as 1./(norm(A_PN*S).^2)
    end
end


% Plots
figure
subplot(1,2,1)
contour(thetas_,phis_,MVDR.')
title('MVDR Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(1,2,2)
surf(thetas_,phis_,MVDR.')
title('MVDR Spectrum Surface Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

figure
subplot(1,2,1)
contour(thetas_,phis_,MUSIC.')
title('MUSIC Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(1,2,2)
surf(thetas_,phis_,MUSIC.')
title('MUSIC Spectrum Surface Plot')
xlabel('theta')
ylabel('phi')


%% (C.7) 
% Both the MVDR and MUSIC spectrum are defined by 1/value (1/norm(R^-0.5*s)^2)
% and (1/norm(P_n*s)^2) so minimization is equivalent to maximizing the 
% denominators. The spectral norm of a matrix is defined as the max x!=0 |Ax|/|x|. 
% Because the s(theta) can be any unit vector,this reduces to the spectral 
% norm of R^-1 for MVDR and P_N for MUSIC. The spectral norm is the largest 
% singular value, which then must be squared.

min_MVDR = 1/((svds((R^-0.5),1)).^2)
min_MUSIC = 1/((svds(A_PN,1)).^2)

%% (C.8) Plug source steering vectors into spectra
s1_MVDR = 1./(norm(R^-0.5*s1).^2)
s1_MUSIC = 1./(norm(A_PN*s1).^2)

s2_MVDR = 1./(norm(R^-0.5*s2).^2)
s2_MUSIC = 1./(norm(A_PN*s2).^2)

s3_MVDR = 1./(norm(R^-0.5*s3).^2)
s3_MUSIC = 1./(norm(A_PN*s3).^2)


%% (C.9) Plug source steering vectors into spectra
MVDR_min_empir = min(min(MVDR))
MUSIC_min_empir = min(min(MUSIC))

% The empirical minimum exactly match the lower bounds found in part7

%% (C.10) Plug source steering vectors into spectra
% Recompute all for the following cases:
% (a) N = 50
% (b) N = 20
% (c) theta2,phi2 = 10

%% N=50
% Number of snapshots
N = 50;

% Parameters (not changing)
d_over_lambda = 0.5;
noise_var = 0.01;    
sig_var = [sig_var_1; sig_var_2; sig_var_3];
thetas = [10;20;30];
phis = [20;-20;150];

% A matrix 
[S_n50,A_n50] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,noise_var,sig_var);
% Theoretical and Approximate Correlation Matrix R of u[n]
[R_n50,R_hat_n50] = cor(A_n50,S_n50,noise_var,sig_var);

% Eigenanalysis --> still can detect the three significant singular values
[right_vecs_n50,eigvals_n50,lft_vecs_n50] = eig(R_n50);
figure
subplot(3,1,1)
stem(abs(diag(eigvals)))
title('N=50 Theoretical R Matrix Singular Values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% Projection Matrix
Q_n50 = lft_vecs_n50(:,1:3);
P_S_n50 = Q_n50*Q_n50'
P_N_n50 = eye(41) - P_S
% Check that the singular vectors lie in the signal space
s1_n50 = S_n50(:,1);
proj_noise_s1_n50=sum(abs(P_N_n50*s1_n50)) %Value is 4.68e-15 ~ 0 --> same as N=100
s2_n50 = S_n50(:,2);
proj_noise_s2_n50=sum(abs(P_N_n50*s2_n50)) %Value is 4.54e-15 ~ 0 --> same as N=100
s3_n50 = S_n50(:,3);
proj_noise_s3_n50=sum(abs(P_N_n50*s3_n50)) %Value is 5.08e-15 ~ 0 --> same as N=100

% Noise-free --> same as N=100
[S_0_n50,A_0_n50] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,0,sig_var)
[R0_n50,R0_hat_n50] = cor(A_0_n50,S_0_n50,0,sig_var) % R and R0 are identical except along the diagonal

% Eigenanalysis of R0
[right_vecs_0_n50,eigvals_0_n50,lft_vecs_0_n50] = eig(R0_n50);
subplot(3,1,2)
stem(abs(diag(eigvals_0_n50)))
title('N=50 No Noise R Matrix Singular Values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% SVD Analysis on A
A_SVD_n50 = svd(A_n50)
subplot(3,1,3)
stem(1:length(A_SVD_n50),A_SVD_n50)
title('N=50 Singular Values of A data matrix')
ylabel('Magnitude of Singular Values')
hold on
plot(repmat(3.5,length(A_SVD_n50),1),linspace(0,10,length(A_SVD_n50)),'--')

% SNR
A_S4_n50 = (A_SVD_n50(4)).^2
noise_var

sum_signal_pwr_n50 = sum(sig_var) 
sum_3_singularvals_n50 = sum((A_SVD_n50(1:3)).^2) 

SNR_n50 = sum_signal_pwr_n50./noise_var
SNR_SVD_n50 = sum_3_singularvals_n50./A_S4_n50

% Spectral Norm
error_n50 = R_n50-R_hat_n50;
outprod_error_n50 = error_n50*error_n50';
[~,error_eigs_n50,~] = eig(outprod_error_n50);
spec_norm_error_n50 = max(max((error_eigs_n50).^0.5))
l2_error_n50 = norm(error_n50)' 

% MVDR and MUSIC Spectrum
[A_U_n50,A_SV_n50,~] = svds(A_n50,3);
A_PS_n50 = A_U_n50*A_U_n50';
A_PN_n50 = eye(41) - A_PS_n50;

thetas = 0:2:90;
phis = 0:2:360;
[thetas_,phis_] = meshgrid(thetas,phis);

MVDR_n50=zeros(46,181);
MUSIC_n50=zeros(46,181);
for t=1:46
    for p=1:181
        S_n50 = steeringvec(thetas(t),phis(p)).';
        MVDR_n50(t,p) = 1./(norm(R_n50^-0.5*S_n50).^2); %same as 1./(S'*inv(R_hat)*S)
        MUSIC_n50(t,p) = 1./(norm(A_PN_n50*S_n50).^2); %same as 1./(norm(A_PN*S).^2)
    end
end

figure
subplot(2,2,1)
contour(thetas_,phis_,MVDR_n50.')
title('N=50 MVDR Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,2)
surf(thetas_,phis_,MVDR_n50.')
title('N=50 MVDR Spectrum Surface Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,3)
contour(thetas_,phis_,MUSIC_n50.')
title('N=50 MUSIC Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,4)
surf(thetas_,phis_,MUSIC_n50.')
title('N=50 MUSIC Spectrum Surface Plot')
xlabel('theta')
ylabel('phi')

% Minimum
min_MVDR_n50 = 1/((svds((R_n50^-0.5),1)).^2)
min_MUSIC_n50 = 1/((svds(A_PN_n50,1)).^2)

s1_MVDR_n50 = 1./(norm(R_n50^-0.5*s1_n50).^2)
s1_MUSIC_n50 = 1./(norm(A_PN_n50*s1_n50).^2)
s2_MVDR_n50 = 1./(norm(R_n50^-0.5*s2_n50).^2)
s2_MUSIC_n50 = 1./(norm(A_PN_n50*s2_n50).^2)
s3_MVDR_n50 = 1./(norm(R_n50^-0.5*s3_n50).^2)
s3_MUSIC_n50 = 1./(norm(A_PN_n50*s3_n50).^2)

MVDR_min_empir_n50 = min(min(MVDR_n50))
MUSIC_min_empir_n50 = min(min(MUSIC_n50))

%% N=20
% Number of snapshots
N = 20;

% Parameters (not changing)
d_over_lambda = 0.5;
noise_var = 0.01;    
sig_var = [sig_var_1; sig_var_2; sig_var_3];
thetas = [10;20;30];
phis = [20;-20;150];

% A matrix 
[S_n20,A_n20] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,noise_var,sig_var);
% Theoretical and Approximate Correlation Matrix R of u[n]
[R_n20,R_hat_n20] = cor(A_n20,S_n20,noise_var,sig_var);

% Eigenanalysis --> still can detect the three significant singular values
[right_vecs_n20,eigvals_n20,lft_vecs_n20] = eig(R_n20);
figure
subplot(3,1,1)
stem(abs(diag(eigvals)))
title('N=20 Theoretical R Matrix Singular Values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% Projection Matrix
Q_n20 = lft_vecs_n20(:,1:3);
P_S_n20 = Q_n20*Q_n20'
P_N_n20 = eye(41) - P_S
% Check that the singular vectors lie in the signal space
s1_n20 = S_n20(:,1);
proj_noise_s1_n20=sum(abs(P_N_n20*s1_n20)) %Value is 4.68e-15 ~ 0 --> same as N=100
s2_n20 = S_n20(:,2);
proj_noise_s2_n20=sum(abs(P_N_n20*s2_n20)) %Value is 4.54e-15 ~ 0 --> same as N=100
s3_n20 = S_n20(:,3);
proj_noise_s3_n20=sum(abs(P_N_n20*s3_n20)) %Value is 5.08e-15 ~ 0 --> same as N=100

% Noise-free --> same as N=100
[S_0_n20,A_0_n20] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,0,sig_var)
[R0_n20,R0_hat_n20] = cor(A_0_n20,S_0_n20,0,sig_var) % R and R0 are identical except along the diagonal

% Eigenanalysis of R0
[right_vecs_0_n20,eigvals_0_n20,lft_vecs_0_n20] = eig(R0_n20);
subplot(3,1,2)
stem(abs(diag(eigvals_0_n20)))
title('N=20 No Noise R Matrix Singular Values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% SVD Analysis on A
A_SVD_n20 = svd(A_n20)
subplot(3,1,3)
stem(1:length(A_SVD_n20),A_SVD_n20)
title('N=20 Singular Values of A data matrix')
ylabel('Magnitude of Singular Values')
hold on
plot(repmat(3.5,length(A_SVD_n20),1),linspace(0,10,length(A_SVD_n20)),'--')

% SNR
A_S4_n20 = (A_SVD_n20(4)).^2
noise_var

sum_signal_pwr_n20 = sum(sig_var) 
sum_3_singularvals_n20 = sum((A_SVD_n20(1:3)).^2) 

SNR_n20 = sum_signal_pwr_n20./noise_var
SNR_SVD_n20 = sum_3_singularvals_n20./A_S4_n20

% Spectral Norm
error_n20 = R_n20-R_hat_n20;
outprod_error_n20 = error_n20*error_n20';
[~,error_eigs_n20,~] = eig(outprod_error_n20);
spec_norm_error_n20 = max(max((error_eigs_n20).^0.5))
l2_error_n20 = norm(error_n20)' 

% MVDR and MUSIC Spectrum
[A_U_n20,A_SV_n20,~] = svds(A_n20,3);
A_PS_n20 = A_U_n20*A_U_n20';
A_PN_n20 = eye(41) - A_PS_n20;

thetas = 0:2:90;
phis = 0:2:360;
[thetas_,phis_] = meshgrid(thetas,phis);

MVDR_n20=zeros(46,181);
MUSIC_n20=zeros(46,181);
for t=1:46
    for p=1:181
        S_n20 = steeringvec(thetas(t),phis(p)).';
        MVDR_n20(t,p) = 1./(norm(R_n20^-0.5*S_n20).^2); %same as 1./(S'*inv(R_hat)*S)
        MUSIC_n20(t,p) = 1./(norm(A_PN_n20*S_n20).^2); %same as 1./(norm(A_PN*S).^2)
    end
end

figure
subplot(2,2,1)
contour(thetas_,phis_,MVDR_n20.')
title('N=20 MVDR Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,2)
surf(thetas_,phis_,MVDR_n20.')
title('N=20 MVDR Spectrum Surface Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,3)
contour(thetas_,phis_,MUSIC_n20.')
title('N=20 MUSIC Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,4)
surf(thetas_,phis_,MUSIC_n20.')
title('N=20 MUSIC Spectrum Surface Plot')
xlabel('theta')
ylabel('phi')

% Minimum
min_MVDR_n20 = 1/((svds((R_n20^-0.5),1)).^2)
min_MUSIC_n20 = 1/((svds(A_PN_n20,1)).^2)

s1_MVDR_n20 = 1./(norm(R_n20^-0.5*s1_n20).^2)
s1_MUSIC_n20 = 1./(norm(A_PN_n20*s1_n20).^2)
s2_MVDR_n20 = 1./(norm(R_n20^-0.5*s2_n20).^2)
s2_MUSIC_n20 = 1./(norm(A_PN_n20*s2_n20).^2)
s3_MVDR_n20 = 1./(norm(R_n20^-0.5*s3_n20).^2)
s3_MUSIC_n20 = 1./(norm(A_PN_n20*s3_n20).^2)

MVDR_min_empir_n20 = min(min(MVDR_n20))
MUSIC_min_empir_n20 = min(min(MUSIC_n20))


%% Theta2,phi2=10
% Number of snapshots
N = 100;

% Parameters (not changing)
d_over_lambda = 0.5;
noise_var = 0.01;    
sig_var = [sig_var_1; sig_var_2; sig_var_3];
thetas = [10;10;30];
phis = [20;10;150];

% A matrix 
[S_ang10,A_ang10] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,noise_var,sig_var);
% Theoretical and Approximate Correlation Matrix R of u[n]
[R_ang10,R_hat_ang10] = cor(A_ang10,S_ang10,noise_var,sig_var);

% Eigenanalysis --> still can detect the three significant singular values
[right_vecs_ang10,eigvals_ang10,lft_vecs_ang10] = eig(R_ang10);
figure
subplot(3,1,1)
stem(abs(diag(eigvals)))
title('Theta2,phi2=10 Theoretical R Matrix Singular Values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% Projection Matrix
Q_ang10 = lft_vecs_ang10(:,1:3);
P_S_ang10 = Q_ang10*Q_ang10'
P_N_ang10 = eye(41) - P_S
% Check that the singular vectors lie in the signal space
s1_ang10 = S_ang10(:,1);
proj_noise_s1_ang10=sum(abs(P_N_ang10*s1_ang10)) %Value is 4.68e-15 ~ 0 --> same as N=100
s2_ang10 = S_ang10(:,2);
proj_noise_s2_ang10=sum(abs(P_N_ang10*s2_ang10)) %Value is 4.54e-15 ~ 0 --> same as N=100
s3_ang10 = S_ang10(:,3);
proj_noise_s3_ang10=sum(abs(P_N_ang10*s3_ang10)) %Value is 5.08e-15 ~ 0 --> same as N=100

% Noise-free
[S_0_ang10,A_0_ang10] = steerANDdata(N,thetas,phis,d_over_lambda,array_locs,0,sig_var)
[R0_ang10,R0_hat_ang10] = cor(A_0_ang10,S_0_ang10,0,sig_var) % R and R0 are identical except along the diagonal

% Eigenanalysis of R0
[right_vecs_0_ang10,eigvals_0_ang10,lft_vecs_0_ang10] = eig(R0_ang10);
subplot(3,1,2)
stem(abs(diag(eigvals_0_ang10)))
title('Theta2,phi2=10 No Noise R Matrix Singular Values')
xlabel('Singular Value Index')
ylabel('Singular Values')

% SVD Analysis on A
A_SVD_ang10 = svd(A_ang10)
subplot(3,1,3)
stem(1:length(A_SVD_ang10),A_SVD_ang10)
title('Theta2,phi2=10 Singular Values of A data matrix')
ylabel('Magnitude of Singular Values')
hold on
plot(repmat(3.5,length(A_SVD_ang10),1),linspace(0,10,length(A_SVD_ang10)),'--')

% SNR
A_S4_ang10 = (A_SVD_ang10(4)).^2
noise_var

sum_signal_pwr_ang10 = sum(sig_var) 
sum_3_singularvals_ang10 = sum((A_SVD_ang10(1:3)).^2) 

SNR_ang10 = sum_signal_pwr_ang10./noise_var
SNR_SVD_ang10 = sum_3_singularvals_ang10./A_S4_ang10

% Spectral Norm
error_ang10 = R_ang10-R_hat_ang10;
outprod_error_ang10 = error_ang10*error_ang10';
[~,error_eigs_ang10,~] = eig(outprod_error_ang10);
spec_norm_error_ang10 = max(max((error_eigs_ang10).^0.5))
l2_error_ang10 = norm(error_ang10)' 

% MVDR and MUSIC Spectrum
[A_U_ang10,A_SV_ang10,~] = svds(A_ang10,3);
A_PS_ang10 = A_U_ang10*A_U_ang10';
A_PN_ang10 = eye(41) - A_PS_ang10;

thetas = 0:2:90;
phis = 0:2:360;
[thetas_,phis_] = meshgrid(thetas,phis);

MVDR_ang10=zeros(46,181);
MUSIC_ang10=zeros(46,181);
for t=1:46
    for p=1:181
        S_ang10 = steeringvec(thetas(t),phis(p)).';
        MVDR_ang10(t,p) = 1./(norm(R_ang10^-0.5*S_ang10).^2); %same as 1./(S'*inv(R_hat)*S)
        MUSIC_ang10(t,p) = 1./(norm(A_PN_ang10*S_ang10).^2); %same as 1./(norm(A_PN*S).^2)
    end
end

figure
subplot(2,2,1)
contour(thetas_,phis_,MVDR_ang10.')
title('Theta2,phi2=10 MVDR Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,2)
surf(thetas_,phis_,MVDR_ang10.')
title('Theta2,phi2=10 MVDR Spectrum Surface Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,3)
contour(thetas_,phis_,MUSIC_ang10.')
title('Theta2,phi2=10 MUSIC Spectrum Contour Plot')
xlabel('Theta Value (degrees)')
ylabel('Phi Value (degrees)')

subplot(2,2,4)
surf(thetas_,phis_,MUSIC_ang10.')
title('Theta2,phi2=10 MUSIC Spectrum Surface Plot')
xlabel('theta')
ylabel('phi')

% Minimum
min_MVDR_ang10 = 1/((svds((R_ang10^-0.5),1)).^2)
min_MUSIC_ang10 = 1/((svds(A_PN_ang10,1)).^2)

s1_MVDR_ang10 = 1./(norm(R_ang10^-0.5*s1_ang10).^2)
s1_MUSIC_ang10 = 1./(norm(A_PN_ang10*s1_ang10).^2)
s2_MVDR_ang10 = 1./(norm(R_ang10^-0.5*s2_ang10).^2)
s2_MUSIC_ang10 = 1./(norm(A_PN_ang10*s2_ang10).^2)
s3_MVDR_ang10 = 1./(norm(R_ang10^-0.5*s3_ang10).^2)
s3_MUSIC_ang10 = 1./(norm(A_PN_ang10*s3_ang10).^2)

MVDR_min_empir_ang10 = min(min(MVDR_ang10))
MUSIC_min_empir_ang10 = min(min(MUSIC_ang10))


%%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function creates an MxN matrix where each entry is a unit variance,
% zero mean complex Gaussian and all entries are independent. The variance
% of n if n = ni + j*nq is additive, var(ni) + var(nq)
%n = @(M,N) 1/sqrt(2)*(randn(M,N) + j*randn(M,N))

% To generate an Mx1 random Gaussian vector with covariance matrix C and mean
% m, start with n, a unit variance white complex Gaussian, C^1.2 is the
% hermitian sq root or cholesky factor
%x = @(C,m,n) (C^0.5)*n + m

%% (A) Function to output steering matrix S and data matrix A
function [S,A]=steerANDdata(N,theta,phi,d_over_lam,r,noise_pwr,source_pwr)
    % Inputs:
    %   1. M = number of sensors in the array
    %   2. N = number of time samples/snapshots
    %   3. theta is a polar angle ranging from 0 to pi/2 and phi is an
    % azimuthal angle ranging from 0 to 2pi. The theta,phi pair form the
    % parameter vector called the angle of arrival (AOA) / direction of
    % arrival (DOA). Theta and phi should each be vectors of dimension Lx1 
    % containing the AOA for each source planewave.
    %   4. lambda is a constant because we only consider the
    % narrowband case.
    %   5. {r_i} i=1...M are the coordinates of the sensor array. Assume
    %   that all coordinates r_i are expressed as multiples of an
    %   underlying spacing d, which must be specified. r_i is a 3d vector
    %   (in order for the dot product between k and r to exist)
    %Lambda and d are wrapped into one constant d/lambda
    %   5. {r_i} i=1...M are the coordinates of the sensor array. Pass in a
    %   3-by-M matrix with the x,y,z coordinates for each sensor in a
    %   column and the various arrays over the rows
    %   6. noise_pwr is the variance of the Gaussian white noise percieved
    %   by the sensor array
    %   7. b is an L-by-N matrix, with time stretching across rows and
    %   source signals down the columns. source_pwr is a L-by-1 vector with
    %   the source variances/powers
    %
    % Outputs:
    %   1. S - steering matrix
    %   2. A - data matrix across time and space
   
    % Number of sources
    [L,~] = size(theta);
    % Number of sensors
    [~,M] = size(r);
    
    % Generate steering matrix
    S = zeros(M,L);
    
    for i=1:L 
        % Unit vector in the direction of the angle of arrival(AOA), 1-by-3
        a_unitvec = [sin(theta(i))*cos(phi(i)) sin(theta(i))*sin(phi(i)) cos(theta(i))];
        
        % Wavenumber vector for a given planewave, 1-by-3
        k = (2*pi./d_over_lam).*a_unitvec;
        
        % Unwrap sensor array coordinates
        %r = r(:); %fix this, 3-by-M, each col is the r vector for a particular sensor
        
        % Steering vector for the particular source
        S(:,i) = (1/sqrt(M)).*exp(-j*k*r);
        % Beta is a matrix with the ns representing time and i looping over l
        % the sources, v is also a matrix so . time goes across the top, each
        % column is a different time.
    end
    
    % Generate data matrix A, whose columns are u[n] for n from 1 to N. 
    % Each u[n] is an M-by-1 snapshot. A is M-by-N.
    
    % Generate Gaussian noise that is white across time and space, M-by-N
    v = (1/sqrt(2))*(randn(M,N)+j*randn(M,N)); %this is unit variance
    v = v.*noise_pwr; %now it has variance=noise_pwr inputed
    
    % b is an L-by-N matrix, with a snapshot of the sources per column and
    % time across the rows
    % Each column in row in b is uncorrelated, white Gaussian across time
    % with variance specified in the input for each source
    b = (1/sqrt(2))*(randn(L,N)+j*randn(L,N)); %this is unit variance
    b = source_pwr.*b; %scale by correposinding signal powers
    
    A = S*b + v;
    
end


%% (B) Function to Calculate Theoretical and Estimated Correlation R Matrix
function [R_theor,R_erg] = cor(A,S,noise_pwr,source_pwr)
    [M,N] = size(A); %num of sensors and time samples
    [~,L] = size(S); %num of sources

    % Theoretical
    %R = S*R_p*S^H + noise_var*I
    R_theor = S*diag(source_pwr)*S' + noise_pwr*eye(M);
    
    % Approximate/Ergodic Correlation - outer product of data A matrix
    R_erg = (1/N).*(A*A'); %A*A^H %add 1/k portion
end

%% (C.6) Function to Calculate Steering Vectors, given AOA, for the particular
% sensor array example described in question C. This function is useful in
% order to compute the MVDR and MUSIC spectrums
function S = steeringvec(theta,phi)
   
    % Parameters defined in problem statement for the particular sensor array
    
    d_over_lambda = 0.5; 
    % Two linear arrays aligned across the x-axis and y-axis. Located at
    % (md,0,0) and (0,md,0) where m spans from -10 to 10.
    m = (-10:10).';
    array_locs = zeros(3,42);
    array_locs(1,1:21) = m; %sensors along x_axis
    array_locs(2,22:42) = m; %sensors along y_axis
    array_locs(:,11) = []; %dont double count origin
    
    %Generate k wavenumber vector, dimension 1-by-3
    k = (2*pi./d_over_lambda).*[sin(theta)*cos(phi) sin(theta)*sin(phi) cos(theta)];
    
    S = (1/sqrt(41)).*exp(-j*k*array_locs);
    % S is normalized to unit length because of the 1/sqrt(41) term
end

