%% Shifra Abittan
% Prof Fontaine
% ECE416 Adaptive Filters
% PSet2 Optimum Filtering

% 1. Haykin Problem 2.18 --> See handwritten

%% 2.
% (A) 
% See handwritten calculation for the derivation of a1 and a2 in the model
% x[n] = v[n] - a1*x[n-1] - a2*x[n-2]
a1 = -1.4;
a2 = 0.48;

% Compute the PSD S(w)
% S(w) for an AR model = var v/magnitude(A(w))^2. Here, v is unit variance
% white noise so var v = 1. S(w) = 1/mag(A(w))^2
% See handwritten calculation.
% S(w) = 1./(1.92*(cosw)^2 - 0.208cosw + 2.2304)

% Compute r[m] (exact values) up to order 10
% r[m] = E(u(n+m)*u*(n))
% r[m] for an AR(2) process is defined as c * s * (-p1/(1-(p1)^2 * p1^(m)
% + p2/(1-(p2)^2 * p2^(m)) where s = sgn(-p1/(1+(p1)^2) + p2/(1+(p2)^2)
p1 = 0.8;
p2 = 0.6;

beta = abs(-p1*(1+p2.^2) + p2*(1+p1^2));
c = 1./beta;
s = sign(-p1./(1+(p1.^2)) + p2./(1+(p2.^2)));

% Anonymous function to compute r[m]
part1 = -p1./(1-(p1.^2));
part2 = p2./(1-(p2.^2));
r_m = @(m) c*s*(part1*(p1^m) + part2*(p2^m)) 

r0 = r_m(0);
r1 = r_m(1);
r2 = r_m(2);
r3 = r_m(3);
r4 = r_m(4);
r5 = r_m(5);
r6 = r_m(6);
r7 = r_m(7);
r8 = r_m(8);
r9 = r_m(9);
r10 = r_m(10);

corr = [r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10]


%% (B) Generate 10^3 samples of x, estimate r[m] up to order 10

% Generate a unit variance white noise signal v of length 10^3 
v = (1/sqrt(2))*(randn(10^3,1));
% Choose first two values of x to begin generation process
x = zeros(10^3,1);
x(1,:) = 0;
x(2,:) = 0;

% Compute x[n] = v[n] + 1.4*x[n-1] - 0.48*x[n-2]
for i=3:10^3
    x(i,:) = v(i,:) + 1.4*x(i-1,:) - 0.48*x(i-2,:);
end 

% Estimate r[m] up to order 10
% Correlation r[m] = E[x(n+m)*conj(x(n))] where m corresponds to the time 
% lag
r0_est = corr_lag(x,0);
r1_est = corr_lag(x,1);
r2_est = corr_lag(x,2);
r3_est = corr_lag(x,3);
r4_est = corr_lag(x,4);
r5_est = corr_lag(x,5);
r6_est = corr_lag(x,6);
r7_est = corr_lag(x,7);
r8_est = corr_lag(x,8);
r9_est = corr_lag(x,9);
r10_est = corr_lag(x,10);

% Correlation r[m] = E[x(n+m)*conj(x(n))] where m corresponds to the time 
% lag 

corr_est = [r0_est r1_est r2_est r3_est r4_est r5_est r6_est r7_est r8_est r9_est r10_est]

% Toeplitz matrices of the correlation values to be used later
R = toeplitz(corr);
R_est = toeplitz(corr_est);


%% (C) Errors between exact and estimate

% Maximum absolute error between the exact and estimated correlation
max_abs_err = max(abs(corr_est - corr)) %find index of this value

% Spectral norm of the difference between the actual correlation matrix and 
% estimated one
% ||R-R_est|| is the spectral norm of the error matrix. For a square matrix,
% the spectral norm is equivalent to the largest singular value or the 
% maximum sqrt(eig(AA^H))
error = corr-corr_est;
outprod_error = error*error';
[~,error_eigs,~] = eig(outprod_error);
spec_norm_error = max(max((error_eigs).^0.5))

l2_error = norm(error)' %Computing the L2 norm returns the same value as 
% when the eigenvalues are used to compute the spectral norm


%% (D) Levinson Durbin algo to compute reflection coefficients

% The reflection coefficients K_m are defined as -delta(m-1) / P(m-1)
% The levinson-durbin algorithm is recursive. It begins with a00 = 1, P0 = r0. 
% Given r0,r1,...rm, we want to find k1, k2, ... km. 
%       Step 1: delta(m-1) = conj(rm) + conj(rm-1)*a(m-1,1) + ... 
%               conj(r1)*a(m-1,m-1) where am0 = 1,...amm = km. 
%       Step 2: km = -delta(m-1) / P(m-1)
%       Step 3: Pm = Pm-1*(1-abs(km)^2)
%       Step 4: Update a_m,k = a_m-1,k + km*a_m-1,m-k

K_est = zeros(10,1); % store the reflection coefficients up to order 10 (k1...k10)
del_est = zeros(11,1); % store the deltas from del_0 to del_10
a_est = zeros(11,11); % store the a vectors for each m
% convention for k>m and k<0 is a_m,k = 0
P_est = zeros(11,1); % store the P values from P_0 to P_10

% Initialize starting values
a_est(1,1) = 1; % a00 = 1
P_est(1) = r0_est;

% m=1
% Inner product of r values in descending order from m to 1 times a vector
del_est(1,1) = conj(r1_est); % del_0 = conj(r1_est)
K_est(1,1) = -del_est(1,1)./P_est(1,1); %-del0/P0
P_est(2,1) = P_est(1,1)*(1-abs(K_est(1,1)).^2); %P1 = P0*(1-|K1|^2)
% a10 = a00 + k1*conj(a01) = a00
% a11 = a01 + k1*conj(a00) = k1
a_est(1:2,2) = [a_est(1,1); K_est(1)];

% m=2 
del_est(2,1) = conj(r2_est) + conj(r1_est)*a_est(2,2); % del_1 = conj(r2_est) + conj(r1_est)*a11
K_est(2,1) = -del_est(2,1)./P_est(2,1); %K2 = -del1/P1
P_est(3,1) = P_est(2,1)*(1-abs(K_est(2,1)).^2); %P2 = P1*(1-|K2|^2)
% a20 = a10 + k2*conj(a12) = a10
% a21 = a11 + k2*conj(a11)
% a22 = a12 + k2*conj(a10) = k2*conj(a10)
a_est(1:3,3) = [a_est(1,2); a_est(1,2)+(K_est(2,1)*conj(a_est(2,2))); K_est(2,1)*conj(a_est(2,2))];

% m=3
for m=3:10
    % Delta Calculation
    descending_r = conj(R_est(1:m,m+1)); %select r_m, r_m-1, ... r1 (not r0!)
    a_m_vec = a_est(1:m,m); % select a_m vector and use values from a0 to am-1 
   
    del_est(m,1) = descending_r.'*a_m_vec; %dot product
    
    % Reflection Coefficient Calculation
    K_est(m,1) = -del_est(m,1)./P_est(m,1);
    
    % P Calculation
    P_est(m+1,1) = P_est(m,1)*(1-abs(K_est(m,1)).^2);
    
    % Update a vector
    a_est(1:m+1,m+1) = [a_est(1:m,m); 0] + K_est(m,1).*[0; flipud(conj(a_est(1:m,m)))];
end  

% Wrapped the above in a function
[K,P,del,a] = levdurb(R,10);

K
K_est

max_err_k = max(abs(K - K_est))

%% (E) Second Order FPEF
% The a coefficients for the 2nd order FPEF can be interpreted as an
% estimate of a1 and a2/pole values. The FPEF for the exact r[m] obviousl
% matches a1 and a2 exactly.
FPEF_2 = a(1:3,3)
% The error for the estimated FPEF:
FPEF_2_est = a_est(1:3,3)
% AR Coefficients
max_err_fpef2_ARcoeff = max(abs(FPEF_2_est(2:3)-[a1;a2]))
% Poles
% To go from the ar coefficients back to poles, solve the system of
% equations: a1 = -(p1+p2); a2 = p1*p2
% p2 = -0.5a1 +/- 0.5*sqrt(a1^2 - 4*a2)
% p1 = -a1-p2
a1_est = FPEF_2_est(2);
a2_est = FPEF_2_est(3);
p2_FPEF_est = -0.5*a1_est + 0.5*sqrt(a1_est^2 - 4*a2_est);
p2_FPEF_est2 = -0.5*a1_est - 0.5*sqrt(a1_est^2 - 4*a2_est);
p1_FPEF_est = -a1_est - p2_FPEF_est;
p1_FPEF_est2 = -a1_est - p2_FPEF_est2;

poles_est = [p1_FPEF_est; p2_FPEF_est]
max_err_fpef2_poles = max(abs([0.8;0.6]-poles_est))

%% (F) The reflection coefficients beyond 2 should be zero
% For the exact case, the K coefficients beyond 2 are all zero besdies for
% numerical imprecision.
K(3:end)
% For the approx case, the K coefficinets are small but not all zero
K_est(3:end)

%% (G)
figure
subplot(1,2,1)
stem(0:10,P)
title('Actual Powers of the Prediction Errors Pm from the Levinson Durbin Algorithm')
xlabel('m')
ylabel('Power of the Prediction Error')

subplot(1,2,2)
stem(0:10,P_est)
title('Estimated Powers of the Prediction Errors Pm from the Levinson Durbin Algorithm')
xlabel('m')
ylabel('Power of the Prediction Error')

%% (H) 
% According to the Payley-Wiener condition, the whitening and innovations
% filters exist iff the entropy is finite. If the coefficients of u(n) are
% normalized to 1, then the entropy = variance of v = 1 (in our case)
%S_w = @(w) log(1./(1.92.*(cos(w)).^2 - 0.208.*cos(w) + 2.2304)) 
S_w = @(w) log(1./((1-0.8.*exp(-j.*w)).*(1-0.8.*exp(j.*w)).*(1-0.6.*exp(-j.*w)).*(1-0.6.*exp(j.*w))))
PWC = exp((1./(2*pi))*integral(S_w,-pi,pi))

% The PWC = 1 as required

%% (I)
values = zeros(11,1)
for M=1:11
    R_M = R(1:M,1:M)
    values(M) = (1./M).*log(det(R_M))
end

figure
stem(1:11,values)
title('1/M*log(det(R_M)) for M from 1 to 11')
xlabel('M')
% They are strictly decreasing and are converging to a little below -2

%% 3. 
% (A)
% Generate a unit variance white noise signal v of length 10^3 
v = (1/sqrt(2))*(randn(1000,1));
% Choose first values of u to begin generation process
u = zeros(1000,1);
u(1,:) = 1;

% Compute u[n] = -0.5*u[n-1] + v[n] - 0.2*v[n-1]
for i=2:10^3
    u(i,:) = -0.5*u(i-1,:) + v(i,:) - 0.2*v(i-1,:);
end 

% Compute estimated correlation matrix
r0_ar3 = corr_lag(u,0)
r1_ar3 = corr_lag(u,1)
r2_ar3 = corr_lag(u,2)
r3_ar3 = corr_lag(u,3)

Corr_ar3 = toeplitz([r0_ar3 r1_ar3 r2_ar3 r3_ar3])
corr_eigs_ar3 = eig(Corr_ar3) 

% Compute bound on mu for steepest descent approach, if mu < 2/maxeig then
% converges to wiener filter
bound_ar3 = 2./max(corr_eigs_ar3)

Corr_ar32 = toeplitz([r0_ar3 r1_ar3 r2_ar3])
corr_eigs_ar32 = eig(Corr_ar32) 

% Compute bound on mu for steepest descent approach, if mu < 2/maxeig then
% converges to wiener filter
bound_ar32 = 2./max(corr_eigs_ar32)

%% (B) Find the 3rd order FPEF using the levinson durbin algorithm
[K_ar3,P_ar3,del_ar3,a_ar3] = levdurb(Corr_ar3,3);
% Reflection coefficients
K_ar3
% Filter coefficients
% The a vector is related to the filter coefficient weights as follows:
% a = [1 -w1 -w2 ... -wM]. Therefore, the 3 filter coefficients are
w0_ar3 = -1.*a_ar3(2:4,4)

%% (C) Steepest descent 
% For the wiener filter, the steepest descent algorithm is w(n+1) = 
% w(n) - mu*(R*w(n) - p) = (I-mu*R)*w(n) + mu*p, p = r-1 r-2 ... r-M where
% conj(r-M) = rM

% R is the correlation matrix of the input signal u 
% p is the cross correlation vector between input signal u and desired signal, E(u*d)

mu1 = 0.1*bound_ar3
mu2 = 0.5*bound_ar3 
mu3 = 0.9*bound_ar3

R = Corr_ar32;
p = [r1_ar3; r2_ar3; r3_ar3]; 
%R = u*u'
%R_erg = (1/1000)*(u*u') % Approximate/Ergodic Correlation - outer product of data A matrix
  
% Here, p = [r-1 r-2 ... r-M] where each entry is E(u(n-i)u(n))

% mu1
count1 = 0;
w1 = [0; 0; 0]; %initialize to any value and will convg
wt1 = w1;
while sum(abs(w1-w0_ar3) > repmat(10^-3,3,1)) ~= 0
    count1 = count1 + 1;
    w1 = w1 - mu1.*(R*w1-p);
    wt1 = [wt1 w1];
end

% mu2
count2 = 0;
w2 = [0; 0; 0]; %initialize to any value and will convg
wt2 = w2;
while sum(abs(w2-w0_ar3) > repmat(10^-3,3,1)) ~= 0
    count2 = count2 + 1;
    w2 = w2 - mu2.*(R*w2-p);
    wt2 = [wt2 w2];
end

% mu3
count3 = 0;
w3 = [0; 0; 0]; %initialize to any value and will convg
wt3 = w3;
while sum(abs(w3-w0_ar3) > repmat(10^-3,3,1)) ~= 0
    count3 = count3 + 1;
    w3 = w3 - mu3.*(R*w3-p);
    wt3 = [wt3 w3];
end

count1
count2
count3
% As the step size increases, requires less iterations to converge.

%% (D) 

beta1 = 0.2;
beta2 = 0.04;
beta3 = 0.008;

% Run levinson durbin backwards. The beta coefficients are the K values.
P0_3D = r0_ar3;

% First, from P_m get all P_m via P_m=P_m-1(1-K_m^2)
P1_3D = P0_3D*(1-(beta1).^2);
P2_3D = P1_3D*(1-(beta2).^2);
P3_3D = P2_3D*(1-(beta3).^2);
% Because it is an ARMA(1,1) process, it makes sense that the power values
% level off after P1! 

% Next, compute the deltas = delta_m-1 = -K_m*P_m-1
delta0_3D = -beta1*P0_3D;
delta1_3D = -beta2*P1_3D;
delta2_3D = -beta3*P2_3D;

% Now compute the a values:
a_3D = zeros(4,4);
a_3D(1,1) = 1; %a00

% a10 = a00 + k1*conj(a01) = a00
a_3D(2,1) = a_3D(1,1);
% a11 = a01 + k1*conj(a00) = k1*conj(a00)
a_3D(2,2) = beta1*conj(a_3D(1,1));

% a20 = a10 + k2*conj(a12) = a10
a_3D(3,1) = a_3D(2,1);
% a21 = a11 + k2*conj(a11)
a_3D(3,2) = a_3D(2,2) + beta2*conj(a_3D(2,2));
% a22 = a12 + k2*conj(a10) = k2*conj(a10)
a_3D(3,3) = beta2*conj(a_3D(2,1));

% a30 = a20 + k3*conj(a23) = a20
a_3D(4,1) = a_3D(3,1);
% a31 = a21 + k3*conj(a22)
a_3D(4,2) = a_3D(3,2) + beta3*conj(a_3D(3,3));
% a32 = a22 + k3*conj(a21)
a_3D(4,3) = a_3D(3,3) + beta3*conj(a_3D(3,2));
% a33 = a23 + k3*conj(a20) = k3*conj(a20)
a_3D(4,4) = beta3*conj(a_3D(3,1));

% Via reverse levinson durbin, recover the r values
r_3D = [P0_3D 0 0 0];
% r1 = conj(delta0)
r1_3D = conj(delta0_3D);
% r2 = conj(delta1) - r1*conj(a11)
r2_3D = conj(delta1_3D) - r1_3D*conj(a_3D(2,2));
% r3 = conj(delta2) - r2*conj(a21) - r1*conj(a22)
r3_3D = conj(delta2_3D) - r2_3D*conj(a_3D(3,2)) - r1_3D*conj(a_3D(3,3));

r_3D = [P0_3D r1_3D r2_3D r3_3D]
r3_3D

% The value is very different. This approximation method does not yield the
% same results.
%% 4. Sensor Arrays and ideal beamformers

% Problem Setup:
% Parameters
d_over_lambda = 0.5;
M = 20; % num sensors
L = 3; % num sources
r = 0:M-1; %sensor array locations - uniform linear array along z axis 

% The azimuthal angle is not considered in this problem. The array pattern
% and AOA is only a function of theta, ranging from 0 to 180 degrees.

% Three sources
src1_angle = 10;
src2_angle = 30;
src3_angle = 50;

thetas = [src1_angle; src2_angle; src3_angle];

% The source powers are expressed in terms of dB, 
% dB = 10 log10(Power/Ref=1) --> 10^(dB/10) = Power
src1_power = 1;
src2_power = 10^(-0.5);  % 5dB below source 1 --> 10^(-5/10)
src3_power = 0.0316;     % 15dB below source 1 --> 10^(-15/10) = 0.0316
bkgrnd_power = 0.0032;   % 25dB below source 1 --> 10^(-25/10) = 0.0032

src_powers = [src1_power; src2_power; src3_power];

% Steering Matrix
S = zeros(M,L);
for i=1:L 
    % AOA is only a function of theta, dim: 1-by-3 for 3 sources
    a = cos(thetas(i));
    % Wavenumber vector for a given planewave, dim: 1-by-3
    k = (2*pi*d_over_lambda).*a;
    % Steering vector for the particular source
    S(:,i) = (1/sqrt(M)).*exp(-j.*k.*r);
end
    
%% (A) 
% Theoretical R matrix for the environment
% R = S*R_p*S^H + noise_var*I
R_theor = S*diag(src_powers)*S' + bkgrnd_power*eye(M)
    
%% (B) 
% Array pattern is defined as A(theta) = |w^H * s(angle)|^2 where w is the
% beamformer vector and s is the steering vector normalized to unit length

% Theta ranges from 0 to 180 degrees but MATLAB requires radian form
theta_range = (0:180)*(pi./180); 
% Compute steering vector for each angle
a_range = cos(theta_range);
k_range = (2*pi*d_over_lambda).*a_range;
S_range = (1/sqrt(M)).*exp(-j.*k_range.*r');

s1 = S(:,1);
s2 = S(:,2);
s3 = S(:,3);

% For each source, form the distortionless MVDR beamforming vector w, defined as 
% w0 = [1/sH(theta0)*inv(R)*s(theta0)]*Rinv(s(theta0)) where s(theta0) is 
% the primary source steering vector. Then plug in the range of theta
% values into the array pattern function and plot.

% Distortionless primary beamforming vectors
w1_dist = (1./(s1'*inv(R_theor)*s1)).*inv(R_theor)*s1; %source 1
w2_dist = (1./(s2'*inv(R_theor)*s2)).*inv(R_theor)*s2; %source 2
w3_dist = (1./(s3'*inv(R_theor)*s3)).*inv(R_theor)*s3; %source 3

Arraypattern_MVDR1 = abs(w1_dist'*S_range).^2;
Arraypattern_MVDR2 = abs(w2_dist'*S_range).^2;
Arraypattern_MVDR3 = abs(w3_dist'*S_range).^2;

% Form array pattern for GSC beamformer by imposing exact nulls in the
% directions of other sources 
S_GSC1 = [S(:,1) repmat(0,20,2)];
S_GSC2 = [repmat(0,20,1) S(:,2) repmat(0,20,1)];
S_GSC3 = [repmat(0,20,2) S(:,3)];

R_theor_GSC1 = S_GSC1*diag(src_powers)*S_GSC1' + bkgrnd_power*eye(M);
R_theor_GSC2 = S_GSC2*diag(src_powers)*S_GSC2' + bkgrnd_power*eye(M);
R_theor_GSC3 = S_GSC3*diag(src_powers)*S_GSC3' + bkgrnd_power*eye(M);

% Distortionless primary beamforming vectors for GSC
w1_dist_GSC = (1./(s1'*inv(R_theor_GSC1)*s1)).*inv(R_theor_GSC1)*s1; %source 1
w2_dist_GSC = (1./(s2'*inv(R_theor_GSC2)*s2)).*inv(R_theor_GSC2)*s2; %source 2
w3_dist_GSC = (1./(s3'*inv(R_theor_GSC3)*s3)).*inv(R_theor_GSC3)*s3; %source 3

Arraypattern_GSC1 = abs(w1_dist_GSC'*S_range).^2;
Arraypattern_GSC2 = abs(w2_dist_GSC'*S_range).^2;
Arraypattern_GSC3 = abs(w3_dist_GSC'*S_range).^2;

figure
subplot(3,1,1)
plot(0:180,20*log10(Arraypattern_MVDR1))
hold on
title('MVDR Array Pattern for the Source Signal 1, theta=10 deg')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-100 0])

subplot(3,1,2)
plot(0:180,20*log10(Arraypattern_MVDR2))
title('MVDR Array Pattern for the Source Signal 2, theta=30 deg')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-100 0])

subplot(3,1,3)
plot(0:180,20*log10(Arraypattern_MVDR3))
title('MVDR Array Pattern for the Source Signal 3, theta=50 deg')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-100 0])

figure
subplot(3,1,1)
plot(0:180,20*log10(Arraypattern_GSC1))
title('GSC Array Patterns for the Source Signal 1, theta=10 deg')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-100 0])

subplot(3,1,2)
plot(0:180,20*log10(Arraypattern_GSC2))
title('GSC Array Patterns for the Source Signal 2, theta=30 deg')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-100 0])

subplot(3,1,3)
plot(0:180,20*log10(Arraypattern_GSC3))
title('GSC Array Patterns for the Source Signal 3, theta=50 deg')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-100 0])

% Compute MVDR attenuation at interfering source angles
MVDR1_atten_ang30 = 20*log10(Arraypattern_MVDR1(:,31))
MVDR1_atten_ang50 = 20*log10(Arraypattern_MVDR1(:,51))

MVDR2_atten_ang10 = 20*log10(Arraypattern_MVDR2(:,11))
MVDR2_atten_ang50 = 20*log10(Arraypattern_MVDR2(:,51))

MVDR3_atten_ang10 = 20*log10(Arraypattern_MVDR3(:,11))
MVDR3_atten_ang30 = 20*log10(Arraypattern_MVDR3(:,31))


%% (C) 
% A(theta) = |w^H*s(theta)|^2 --> A(w_hat) = |W(w_hat)|^2
% Therefore, take each of the w vectors, of length m, and take them to be
% the FIR filter coefficients. Compute the frequency response and plug in
% angles.

% Electrical angle is w_hat = 2*pi*d/lambda * cos(theta)
full_theta_range = (-180:180).*(pi./180);
d_over_lambda = 0.4; % If less than 0.5, spacial aliasing occurs
w_hat_full = (2*pi*d_over_lambda).*cos(full_theta_range);
S_full_range = (1/sqrt(M)).*exp(-j.*full_theta_range.*r');

% Compute steering vector for each source angle
k_range = (2*pi*0.4).*cos(thetas');
S_src = (1/sqrt(M)).*exp(-j.*k_range.*r');
s1_src= S_src(:,1);
s2_src = S_src(:,2);
s3_src = S_src(:,3);

R_theor_full = S_src*diag(src_powers)*S_src' + bkgrnd_power*eye(M);

% Distortionless primary beamforming vectors
w1_dist_full = (1./(s1_src'*inv(R_theor_full)*s1_src)).*inv(R_theor_full)*s1_src; %source 1
w2_dist_full = (1./(s2_src'*inv(R_theor_full)*s2_src)).*inv(R_theor_full)*s2_src; %source 2
w3_dist_full = (1./(s3_src'*inv(R_theor_full)*s3_src)).*inv(R_theor_full)*s3_src; %source 3

% Take magnitude of frequency response of beamforming vectors
h1 = freqz(conj(w1_dist_full),361,full_theta_range);
h2 = freqz(conj(w2_dist_full),361,full_theta_range);
h3 = freqz(conj(w3_dist_full),361,full_theta_range);

cutoff_visible = 2*pi*0.4;
cutoff_visible_deg = cutoff_visible*(180./pi);

figure
subplot(3,1,1)
plot((full_theta_range).*(180./pi),abs(h1),'r')
hold on
line([cutoff_visible_deg cutoff_visible_deg], [0 0.015]);
hold on
line([-cutoff_visible_deg -cutoff_visible_deg], [0 0.015]);
title('MVDR Source 1: Magnitude Response of FIR filter where coefficients are {w*_m}')
xlabel('w_hat')
ylabel('A(w_hat)')
legend('Mag Response','Visible Region Cutoff','Visible Region Cutoff')

subplot(3,1,2)
plot((full_theta_range).*(180./pi),abs(h2),'r')
hold on
line([cutoff_visible_deg cutoff_visible_deg], [0 0.015]);
hold on
line([-cutoff_visible_deg -cutoff_visible_deg], [0 0.015]);
title('MVDR Source 2: Magnitude Response of FIR filter where coefficients are {w*_m}')
xlabel('w_hat')
ylabel('A(w_hat)')
legend('Mag Response','Visible Region Cutoff','Visible Region Cutoff')

subplot(3,1,3)
plot((full_theta_range).*(180./pi),abs(h3),'r')
hold on
line([cutoff_visible_deg cutoff_visible_deg], [0 0.015]);
hold on
line([-cutoff_visible_deg -cutoff_visible_deg], [0 0.015]);
title('MVDR Source 3: Magnitude Response of FIR filter where coefficients are {w*_m}')
xlabel('w_hat')
ylabel('A(w_hat)')
legend('Mag Response','Visible Region Cutoff','Visible Region Cutoff')

% Peak Values
MVDR1_max_invis = max([abs(h1(:,1:37)) abs(h1(:,325:end))])
MVDR2_max_invis = max([abs(h2(:,1:37)) abs(h2(:,325:end))])
MVDR3_max_invis = max([abs(h3(:,1:37)) abs(h3(:,325:end))])

R_theor_full_GSC1 = S_GSC1*diag(src_powers)*S_GSC1' + bkgrnd_power*eye(M);
R_theor_full_GSC2 = S_GSC2*diag(src_powers)*S_GSC2' + bkgrnd_power*eye(M);
R_theor_full_GSC3 = S_GSC3*diag(src_powers)*S_GSC3' + bkgrnd_power*eye(M);

% Distortionless primary beamforming vectors
w1_dist_full_GSC = (1./(s1_src'*inv(R_theor_full_GSC1)*s1_src)).*inv(R_theor_full_GSC1)*s1_src; %source 1
w2_dist_full_GSC = (1./(s2_src'*inv(R_theor_full_GSC2)*s2_src)).*inv(R_theor_full_GSC2)*s2_src; %source 2
w3_dist_full_GSC = (1./(s3_src'*inv(R_theor_full_GSC3)*s3_src)).*inv(R_theor_full_GSC3)*s3_src; %source 3

% Take magnitude of frequency response of beamforming vectors
h1_GSC = freqz(conj(w1_dist_full_GSC),361,full_theta_range);
h2_GSC = freqz(conj(w2_dist_full_GSC),361,full_theta_range);
h3_GSC = freqz(conj(w3_dist_full_GSC),361,full_theta_range);

cutoff_visible = 2*pi*0.4;
cutoff_visible_deg = cutoff_visible*(180./pi);

figure
subplot(3,1,1)
plot((full_theta_range).*(180./pi),abs(h1_GSC),'r')
hold on
line([cutoff_visible_deg cutoff_visible_deg], [0 0.015]);
hold on
line([-cutoff_visible_deg -cutoff_visible_deg], [0 0.015]);
title('GSC Source 1: Magnitude Response of FIR filter where coefficients are {w*_m}')
xlabel('w_hat')
ylabel('A(w_hat)')
legend('Mag Response','Visible Region Cutoff','Visible Region Cutoff')

subplot(3,1,2)
plot((full_theta_range).*(180./pi),abs(h2_GSC),'r')
hold on
line([cutoff_visible_deg cutoff_visible_deg], [0 0.015]);
hold on
line([-cutoff_visible_deg -cutoff_visible_deg], [0 0.015]);
title('GSC Source 2: Magnitude Response of FIR filter where coefficients are {w*_m}')
xlabel('w_hat')
ylabel('A(w_hat)')
legend('Mag Response','Visible Region Cutoff','Visible Region Cutoff')

subplot(3,1,3)
plot((full_theta_range).*(180./pi),abs(h3_GSC),'r')
hold on
line([cutoff_visible_deg cutoff_visible_deg], [0 0.015]);
hold on
line([-cutoff_visible_deg -cutoff_visible_deg], [0 0.015]);
title('GSC Source 3: Magnitude Response of FIR filter where coefficients are {w*_m}')
xlabel('w_hat')
ylabel('A(w_hat)')
legend('Mag Response','Visible Region Cutoff','Visible Region Cutoff')

% Peak Values
MVDR1_max_invis = max([abs(h1(:,1:37)) abs(h1(:,325:end))])
MVDR2_max_invis = max([abs(h2(:,1:37)) abs(h2(:,325:end))])
MVDR3_max_invis = max([abs(h3(:,1:37)) abs(h3(:,325:end))])




%%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% (2.b) Function to estimate r[m], autocorrelation with lag m
function r = corr_lag(x,M)
    % Chop off M last values of x(n) because the corresponding x(n+m) are
    % not available
    x_n = x(1:end-M);
    % Remove first M values of x(n+m) because the corresponding x(n) values
    % are not available
    x_nm = x(M+1:end); %because MATLAB indexes from 1, must add 1 to M
    
    r = mean(x_n.*x_nm);
end

%% (2.d) Levinson Durbin Algorithm Implementation
function [K,P,del,a] = levdurb(R,order)

    % The reflection coefficients K_m are defined as -delta(m-1) / P(m-1)
    % The levinson-durbin algorithm is recursive. It begins with a00 = 1, P0 = r0. 
    % Given r0,r1,...rm, we want to find k1, k2, ... km. 
    %       Step 1: delta(m-1) = conj(rm) + conj(rm-1)*a(m-1,1) + ... 
    %               conj(r1)*a(m-1,m-1) where am0 = 1,...amm = km. 
    %       Step 2: km = -delta(m-1) / P(m-1)
    %       Step 3: Pm = Pm-1*(1-abs(km)^2)
    %       Step 4: Update a_m,k = a_m-1,k + km*a_m-1,m-k

    K = zeros(order,1); % store the reflection coefficients up to order 10 (k1...k10)
    del = zeros(order+1,1); % store the deltas from del_0 to del_10
    a = zeros(order+1,order+1); % store the a vectors for each m
    % convention for k>m and k<0 is a_m,k = 0
    P = zeros(order+1,1); % store the P values from P_0 to P_10

    % Initialize starting values
    a(1,1) = 1; % a00 = 1
    P(1) = R(1,1);

    for m = 1:order
        % Delta Calculation
        descending_r = conj(R(1:m,m+1)); %select r_m, r_m-1, ... r1 (not r0!)
        a_m_vec = a(1:m,m); % select a_m vector and use values from a0 to am-1 

        del(m,1) = descending_r.'*a_m_vec; %dot product

        % Reflection Coefficient Calculation
        K(m,1) = -del(m,1)./P(m,1);

        % P Calculation
        P(m+1,1) = P(m,1)*(1-abs(K(m,1)).^2);

        % Update a vector
        a(1:m+1,m+1) = [a(1:m,m); 0] + K(m,1).*[0; flipud(conj(a(1:m,m)))];
    end  
end