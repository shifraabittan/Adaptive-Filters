%% Shifra Abittan
% Prof Fontaine
% ECE416 Adaptive Filters
% PSet3 LMS and NLMS

%% 1. 
% For the signal u[n], assume an AR model: u[n]=v1[n] - a1*u[n-1] -
% a2*u[n-2]. v1 is unit variance white complex Gaussian noise. To compute
% the a1,a2 values from the poles in an AR[2] model, a1 = -(p1+p2) and a2 =
% p1*p2

% Case 1: poles are at 0.3 and 0.5
c1_p1 = 0.3;
c1_p2 = 0.5;
% a1,a2 values
c1_a1 = -(c1_p1+c1_p2);
c1_a2 = c1_p1*c1_p2;

% Case 2: poles are at 0.3 and 0.95
c2_p1 = 0.3;
c2_p2 = 0.95;
% a1,a2 values
c2_a1 = -(c2_p1+c2_p2);
c2_a2 = c2_p1*c2_p2;

% Three different model orders will be examined:
M1 = 3;
M2 = 6;
M3 = 10;

% Beta is a vector of length 6, defined as b_0k = 1./k^2 for k from 1 to 6
beta_0 = 1./((1:6).^2);

%% Generate u and d signals to be used in part B
% Unit variance, complex white noise signal v of length 10^3 
v1 = (1/sqrt(2))*(randn(10^3,1) + j*randn(10^3,1));
v2 = (1/sqrt(2))*(randn(10^3,1) + j*randn(10^3,1));

% Case 1:
u1 = zeros(10^3,1);
% Compute u[3]-u[6] because loop cant begin until n=7 due to constraints on
% d[n] equation (explained below).
u1(3,:) = v1(3,:) - c1_a1*u1(2,:) - c1_a2*u1(1,:);
u1(4,:) = v1(4,:) - c1_a1*u1(3,:) - c1_a2*u1(2,:);
u1(5,:) = v1(5,:) - c1_a1*u1(4,:) - c1_a2*u1(3,:);
u1(6,:) = v1(6,:) - c1_a1*u1(5,:) - c1_a2*u1(4,:);

d1 = zeros(10^3,1);
% Initialize first 6 values of d1 = v2. To compute d[n], requires last 6
% values of u because beta_0 is of length 6. Can't compute the first term
% in d until n=7.
d1(1:6,:) = v2(1:6,:);

% Compute u[n] = v1[n] - a1*u[n-1] - a2*u[n-2]
for i=6:10^3
    u1(i,:) = v1(i,:) - c1_a1*u1(i-1,:) - c1_a2*u1(i-2,:);
    d1(i,:) = beta_0*flip(u1(i-5:i,:)) + v2(i,:); %u1 must be flipped because 
    % u_m is defined as [u(n), u(n-1),... u(n-M+1)]'
end 

% Case 2:
u2 = zeros(10^3,1);
% Compute u[3]-u[6] because loop cant begin until n=7 due to constraints on
% d[n] equation (explained below).
u2(3,:) = v1(3,:) - c2_a1*u1(2,:) - c2_a2*u2(1,:);
u2(4,:) = v1(4,:) - c2_a1*u1(3,:) - c2_a2*u2(2,:);
u2(5,:) = v1(5,:) - c2_a1*u1(4,:) - c2_a2*u2(3,:);
u2(6,:) = v1(6,:) - c2_a1*u1(5,:) - c2_a2*u2(4,:);

d2 = zeros(10^3,1);
% Initialize first 6 values of d2 = v2. To compute d[n], requires last 6
% values of u because beta_0 is of length 6. Can't compute the first term
% in d until n=7.
d2(1:6,:) = v2(1:6,:);

% Compute u[n] = v1[n] - a1*u[n-1] - a2*u[n-2]
for i=6:10^3
    u2(i,:) = v1(i,:) - c2_a1*u2(i-1,:) - c2_a2*u2(i-2,:);
    d2(i,:) = beta_0*flip(u2(i-5:i,:)) + v2(i,:);
end 


%% (A) Compute theoretical wiener filter coefficients, bounds on LMS and NLMS

% Compute R_m, exactly for each model order, using the coefficient values 
% and not a time average. 

% Case 1
[c1_R3,c1_P3] = exact_Rm_pm(c1_p1,c1_p2,M1,beta_0)
[c1_R6,c1_P6] = exact_Rm_pm(c1_p1,c1_p2,M2,beta_0)
[c1_R10,c1_P10] = exact_Rm_pm(c1_p1,c1_p2,M3,beta_0)

% Case 2
[c2_R3,c2_P3] = exact_Rm_pm(c2_p1,c2_p2,M1,beta_0)
[c2_R6,c2_P6] = exact_Rm_pm(c2_p1,c2_p2,M2,beta_0)
[c2_R10,c2_P10] = exact_Rm_pm(c2_p1,c2_p2,M3,beta_0)


%% Compute the Wiener filter coefficients w0 = inv(R)*p

% Case 1
c1_w03 = inv(c1_R3)*c1_P3
c1_w06 = inv(c1_R6)*c1_P6
c1_w010 = inv(c1_R10)*c1_P10

% Case 2
c2_w03 = inv(c2_R3)*c2_P3
c2_w06 = inv(c2_R6)*c2_P6
c2_w010 = inv(c2_R10)*c2_P10

%% Compute |w_0M - beta|, extending with 0 if necessary

% Case 1
c1_err3 = abs([c1_w03;0;0;0] - beta_0')
c1_err6 = abs(c1_w06 - beta_0')
c1_err10 = abs(c1_w010 - [beta_0 0 0 0 0]')
% Errors very tiny for M=6, M=10 but very large for M=3

% Case 2
c2_err3 = abs([c2_w03;0;0;0] - beta_0')
c2_err6 = abs(c2_w06 - beta_0')
c2_err10 = abs(c2_w010 - [beta_0 0 0 0 0]')
% Errors very tiny for M=6, M=10 but very large for M=3. The overall errors
% are 2 orders of magnitude larger than in Case 1.

%% Eigenspread = max mag eigenvalue/min mag eigenvalue  

% Case 1
c1_R3_eig = abs(eig(c1_R3));
c1_M3_eigenspread = max(c1_R3_eig)./min(c1_R3_eig)

c1_R6_eig = abs(eig(c1_R6));
c1_M6_eigenspread = max(c1_R6_eig)./min(c1_R6_eig)

c1_R10_eig = abs(eig(c1_R10));
c1_M10_eigenspread = max(c1_R10_eig)./min(c1_R10_eig)

% Case 2
c2_R3_eig = abs(eig(c2_R3));
c2_M3_eigenspread = max(c2_R3_eig)./min(c2_R3_eig)

c2_R6_eig = abs(eig(c2_R6));
c2_M6_eigenspread = max(c2_R6_eig)./min(c2_R6_eig)

c2_R10_eig = abs(eig(c2_R10));
c2_M10_eigenspread = max(c2_R10_eig)./min(c2_R10_eig)

% Eigenspread values significantly larger for case 2, when the poles are
% further apart and one is closer to the edge of stability/close to 1.

%% Calculate the bound on mu for a stable LMS algorithm, u_max = 2/eigen_max
% Case 1
c1_M3_u_max = 2./max(c1_R3_eig)
c1_M6_u_max = 2./max(c1_R6_eig)
c1_M10_u_max = 2./max(c1_R10_eig)
% Case 2
c2_M3_u_max = 2./max(c2_R3_eig)
c2_M6_u_max = 2./max(c2_R6_eig)
c2_M10_u_max = 2./max(c2_R10_eig)

% Larger model orders reduce the maximum mu, taking smaller stepsizes

%% Calculate the bound on mu for a stable NLMS algorithm, u_max_tilda
% u_max_tilda = 2*D(n)*E(|u(n)|^2)/E(|e(n)|^2) where E(|u(n)|^2) is the input
% signal power, E(|e(n)|^2) is the error signal power and D(n) is the mean
% squared deviation = E(||curlye(n)||^2 where curlye = w0 - what

% The reason for using NLMS is that u_tilda is adaptively changing. The
% fractional portion of u_max_tilda

input_sig_pwr_c1 = mean(abs(u1).^2); %approx 2
input_sig_pwr_c2 = mean(abs(u2).^2); %approx 20

% Assume that D(n) will be at most the difference between w0 and zero
% vector. This is because the adaptive algorithm should on average be
% improving the weights.
D_c1_M3_bound = mean(abs(c1_w03).^2);
D_c1_M6_bound = mean(abs(c1_w06).^2);
D_c1_M10_bound = mean(abs(c1_w010).^2);

D_c2_M3_bound = mean(abs(c2_w03).^2);
D_c2_M6_bound = mean(abs(c2_w06).^2);
D_c2_M10_bound = mean(abs(c2_w010).^2);
% approx between 0.1 and 0.3

% Thus far, we have approx 2*0.1 - 2*0.3 = 0.2 to 0.6 for case 1 and 
% 20*0.1 - 20*0.3 = 2 to 6 for case 2. These values get multiplied by the
% constant 2 in the front --> Case1: 0.4 to 1.2 and Case2: 4 to 12. Then
% divide by some decimal value because the a values are >1. This will
% increase the approximations.  Therefore, a guess of 2 as an upper bound
% is reasonable.

u_max_tilda = 2;



%% (B) Run LMS (3 step sizes) and NLMS (1 step size) and generate J(n) 
% learning curve and D(n) mean-square deviation curve by averaging over 100
% runs. Do this for all 6 cases.

% The formula for LMS is w(n+1) = w(n) + mu*u_M(n)*conj(e(n)) where
% conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H)

% CASE 1, M=3, LMS Stepsize = 0.05*u_max
mu = 0.05*c1_M3_u_max;

len = 250;
J = zeros(100,len); 
D = zeros(100,len);

% Perform 100 runs to generate reasonable, non-noisy learning curves
for runs = 1:100
    % Generate fresh u and d for each run
    [u1,d1] = u_and_d(len,1);
    
    % Initialize w
    w_LMS1_c1_M3 = [0; 0; 0];
    % Keep track of w progression
    recw_LMS1_c1_M3 = w_LMS1_c1_M3;

    M=3;

    for itr = 7:len
        
        % y = w^H*u
        y = w_LMS1_c1_M3'*flipud(u1(itr-(M-1):itr,1)); % When pulling u values, make sure to flip vertically
        
        % err = d - y
        % conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H
        err = (d1(itr,1) - y)';

        % w(n+1) = w(n) + mu*u_M(n)*conj(e(n))
        w_LMS1_c1_M3 = w_LMS1_c1_M3 + mu*flipud(u1(itr-(M-1):itr,1))*err;
        recw_LMS1_c1_M3 = [recw_LMS1_c1_M3 w_LMS1_c1_M3];

        % MSE Learning Curve J(n) = E(|e(n)|^2)
        J(runs,itr-6) = abs(d1(itr,1) - y).^2;
        % Mean Square Deviation Learning Curve D(n) = E(||w0-what||^2)
        D(runs,itr-6) = norm(c1_w03-w_LMS1_c1_M3).^2;
    end
end

% Average over 100 runs
J_c1_LMS1_M3 = mean(J);
D_c1_LMS1_M3 = mean(D);

% Graph
figure
sgtitle('Pole Case 1, M=3')
subplot(4,2,1)
plot(1:150,J_c1_LMS1_M3(1:150))
title('LMS J Learning Curve for u = 0.05*umax')
xlabel('n (iteration)')
ylabel('E[ | e(n) |^2 ]')

subplot(4,2,2)
plot(1:150,D_c1_LMS1_M3(1:150))
title('LMS D Learning Curve for u = 0.05*umax')
xlabel('n (iteration)')
ylabel('E[ ||w0 - wapprox||^2 ]')



% For other 17 LMS cases, use function that wraps up above code. 

% CASE 1, M=3, LMS Stepsize = 0.5*u_max
mu = 0.5*c1_M3_u_max;
M = 3;
case_num = 1;
w0 = c1_w03;
numruns = 100;
len = 250;

[J_c1_LMS2_M3,D_c1_LMS2_M3] = LMS(mu,M,case_num,w0,numruns,len);

% Graph
subplot(4,2,3)
plot(1:150,J_c1_LMS2_M3(1:150))
title('LMS J Learning Curve for u = 0.5*umax')
xlabel('n (iteration)')
ylabel('E[ | e(n) |^2 ]')

subplot(4,2,4)
plot(1:150,D_c1_LMS2_M3(1:150))
title('LMS D Learning Curve for u = 0.5*umax')
xlabel('n (iteration)')
ylabel('E[ ||w0 - wapprox||^2 ]')

% CASE 1, M=3, LMS Stepsize = 0.8*u_max
mu = 0.8*c1_M3_u_max;
[J_c1_LMS3_M3,D_c1_LMS3_M3] = LMS(mu,3,1,c1_w03,100,len);

% Graph
subplot(4,2,5)
plot(1:150,J_c1_LMS3_M3(1:150))
title('LMS J Learning Curve for u = 0.8*umax')
xlabel('n (iteration)')
ylabel('E[ | e(n) |^2 ]')

subplot(4,2,6)
plot(1:150,D_c1_LMS3_M3(1:150))
title('LMS D Learning Curve for u = 0.8*umax')
xlabel('n (iteration)')
ylabel('E[ ||w0 - wapprox||^2 ]')

% The filter is no longer stable once the mu gets too large.

% NLMS Case 1, M=3

% The formula for NLMS is w(n+1) = w(n) + mu_tilda*(1./lamd+||u_M||^2)*u_M(n)*conj(e(n))) 
% where conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H) and lamd is greater than 0 but
% small

% NLMS Stepsize = 0.2*mu_tilda_max

len = 1000;
J = zeros(100,len); 
D = zeros(100,len);

% Perform 100 runs to generate reasonable, non-noisy learning curves
for runs = 1:100
    % Generate fresh u and d for each run
    [u1,d1] = u_and_d(len,1);
    
    % Initialize w
    w_NLMS_c1_M3 = [0; 0; 0];
    % Keep track of w progression
    recw_NLMS_c1_M3 = w_NLMS_c1_M3;

    M=3;
    lamd = 0.0001;
    
    for itr = 7:len
        
        % y = w^H*u
        y = w_NLMS_c1_M3'*flipud(u1(itr-(M-1):itr,1)); % When pulling u values, make sure to flip vertically
        
        % e = d - y
        e = d1(itr,1) - y;
        
        % mu_tilda_max
        mu = 0.2*mu_tilda_max;

        % w(n+1) = w(n) + mu*u_M(n)*conj(e(n))
        % w(n+1) = w(n) + mu_tilda*(1./lamd+||u_M||^2)*u_M(n)*conj(e(n)))
        constants = mu./(lamd + norm(u1(itr-(M-1):itr,1)).^2);
        w_NLMS_c1_M3 = w_NLMS_c1_M3 + constants*e'*flipud(u1(itr-(M-1):itr,1));
        recw_NLMS_c1_M3 = [recw_NLMS_c1_M3 w_NLMS_c1_M3];

        % MSE Learning Curve J(n) = E(|e(n)|^2)
        J(runs,itr-6) = abs(d1(itr,1) - y).^2;
        % Mean Square Deviation Learning Curve D(n) = E(||w0-what||^2)
        D(runs,itr-6) = norm(c1_w03-w_NLMS_c1_M3).^2;
    end
end

% Average over 100 runs
J_c1_NLMS_M3 = mean(J);
D_c1_NLMS_M3 = mean(D);


% Graph
subplot(4,2,7)
plot(1:150,J_c1_NLMS_M3(1:150))
title('NLMS J Learning Curve for u = 0.2*umax_tilda')
xlabel('n (iteration)')
ylabel('E[ | e(n) |^2 ]')

subplot(4,2,8)
plot(1:150,D_c1_NLMS_M3(1:150))
title('NLMS D Learning Curve for u = 0.2*umax_tilda')
xlabel('n (iteration)')
ylabel('E[ ||w0 - wapprox||^2 ]')

% The D graph is scaled by approximately 1/4 of the J graph, but they do
% track each other as expected.

%% Poles Case 1, M=6

% LMS Stepsize = 0.05*u_max
mu = 0.05*c1_M6_u_max;
[J_c1_LMS1_M6,D_c1_LMS1_M6] = LMS(mu,6,1,c1_w06,100,len);
% LMS Stepsize = 0.5*u_max
mu = 0.5*c1_M6_u_max;
[J_c1_LMS2_M6,D_c1_LMS2_M6] = LMS(mu,6,1,c1_w06,100,len);
% LMS Stepsize = 0.8*u_max
mu = 0.8*c1_M6_u_max;
[J_c1_LMS3_M6,D_c1_LMS3_M6] = LMS(mu,6,1,c1_w06,100,len);
% NLMS
[J_c1_NLMS_M6,D_c1_NLMS_M6] = NLMS(mu,6,1,c1_w06,100,len,0.001);

% Function to wrap plotting code
graphDJ('Pole Case 1, M=6',len,J_c1_LMS1_M6,D_c1_LMS1_M6,J_c1_LMS2_M6,D_c1_LMS2_M6,J_c1_LMS3_M6,D_c1_LMS3_M6,J_c1_NLMS_M6,D_c1_NLMS_M6)


%% Case 1, M=10

% LMS Stepsize = 0.05*u_max
mu = 0.05*c1_M10_u_max;
[J_c1_LMS1_M10,D_c1_LMS1_M10] = LMS(mu,10,1,c1_w010,100,len);
% LMS Stepsize = 0.5*u_max
mu = 0.5*c1_M10_u_max;
[J_c1_LMS2_M10,D_c1_LMS2_M10] = LMS(mu,10,1,c1_w010,100,len);
% LMS Stepsize = 0.8*u_max
mu = 0.8*c1_M10_u_max;
[J_c1_LMS3_M10,D_c1_LMS3_M10] = LMS(mu,10,1,c1_w010,100,len);
% NLMS
[J_c1_NLMS_M10,D_c1_NLMS_M10] = NLMS(mu,10,1,c1_w010,100,len,0.001);

% Function to wrap plotting code
graphDJ('Pole Case 1, M=10',len,J_c1_LMS1_M10,D_c1_LMS1_M10,J_c1_LMS2_M10,D_c1_LMS2_M10,J_c1_LMS3_M10,D_c1_LMS3_M10,J_c1_NLMS_M10,D_c1_NLMS_M10)


%% Poles Case 2, M=3

% LMS Stepsize = 0.05*u_max
mu = 0.05*c2_M3_u_max;
[J_c2_LMS1_M3,D_c2_LMS1_M3] = LMS(mu,3,2,c2_w03,100,len);
% LMS Stepsize = 0.5*u_max
mu = 0.5*c2_M3_u_max;
[J_c2_LMS2_M3,D_c2_LMS2_M3] = LMS(mu,3,2,c2_w03,100,len);
% LMS Stepsize = 0.8*u_max
mu = 0.8*c2_M3_u_max;
[J_c2_LMS3_M3,D_c2_LMS3_M3] = LMS(mu,3,2,c2_w03,100,len);
% NLMS
[J_c2_NLMS_M3,D_c2_NLMS_M3] = NLMS(mu,3,2,c2_w03,100,len,0.001);

% Function to wrap plotting code
graphDJ('Pole Case 2, M=3',len,J_c2_LMS1_M3,D_c2_LMS1_M3,J_c2_LMS2_M3,D_c2_LMS2_M3,J_c2_LMS3_M3,D_c2_LMS3_M3,J_c2_NLMS_M3,D_c2_NLMS_M3)


%% Poles Case 2, M=6

% LMS Stepsize = 0.05*u_max
mu = 0.05*c2_M6_u_max;
[J_c2_LMS1_M6,D_c2_LMS1_M6] = LMS(mu,6,2,c2_w06,100,len);
% LMS Stepsize = 0.5*u_max
mu = 0.5*c2_M6_u_max;
[J_c2_LMS2_M6,D_c2_LMS2_M6] = LMS(mu,6,2,c2_w06,100,len);
% LMS Stepsize = 0.8*u_max
mu = 0.8*c2_M6_u_max;
[J_c2_LMS3_M6,D_c2_LMS3_M6] = LMS(mu,6,2,c2_w06,100,len);
% NLMS
%mu =
[J_c2_NLMS_M6,D_c2_NLMS_M6] = NLMS(mu,6,2,c2_w06,100,len,0.001);

% Function to wrap plotting code
graphDJ('Pole Case 2, M=6',len,J_c2_LMS1_M6,D_c2_LMS1_M6,J_c2_LMS2_M6,D_c2_LMS2_M6,J_c2_LMS3_M6,D_c2_LMS3_M6,J_c2_NLMS_M6,D_c2_NLMS_M6)


%% Pole Case 2, M=10

% LMS Stepsize = 0.05*u_max
mu = 0.05*c2_M10_u_max;
[J_c2_LMS1_M10,D_c2_LMS1_M10] = LMS(mu,10,2,c2_w010,100,len);
% LMS Stepsize = 0.5*u_max
mu = 0.5*c2_M10_u_max;
[J_c2_LMS2_M10,D_c2_LMS2_M10] = LMS(mu,10,2,c2_w010,100,len);
% LMS Stepsize = 0.8*u_max
mu = 0.8*c2_M10_u_max;
[J_c2_LMS3_M10,D_c2_LMS3_M10] = LMS(mu,10,2,c2_w010,100,len);
% NLMS
%mu =
[J_c2_NLMS_M10,D_c2_NLMS_M10] = NLMS(mu,10,2,c2_w010,100,len,0.001);

% Function to wrap plotting code
graphDJ('Pole Case 2, M=10',len,J_c2_LMS1_M10,D_c2_LMS1_M10,J_c2_LMS2_M10,D_c2_LMS2_M10,J_c2_LMS3_M10,D_c2_LMS3_M10,J_c2_NLMS_M10,D_c2_NLMS_M10)


%% Compare Models Case 1
figure
subplot(1,2,1)
title('J Learning Curves for u = 0.05*umax for case 1 as M varies')
plot(1:100,J_c1_LMS1_M3(1:100))
hold on
plot(1:100,J_c1_LMS1_M6(1:100))
hold on
plot(1:100,J_c1_LMS1_M10(1:100))
hold on
title('J Learning Curves for u = 0.05*umax for case 1 as M varies')
legend('M3','M6','M10')
xlabel('n (iteration)')
ylabel('E[ | e(n) |^2 ]')

subplot(1,2,2)
title('D Learning Curves for u = 0.05*umax for case 1 as M varies')
plot(1:100,D_c1_LMS1_M3(1:100))
hold on
plot(1:100,D_c1_LMS1_M6(1:100))
hold on
plot(1:100,D_c1_LMS1_M10(1:100))
hold on
title('D Learning Curves for u = 0.05*umax for case 1 as M varies')
legend('M3','M6','M10')
xlabel('n (iteration)')
ylabel('E[ | curlye(n) |^2 ]')


%% Compare Models Case 2
figure
subplot(1,2,1)
plot(1:400,J_c2_LMS1_M3(1:400))
hold on
plot(1:400,J_c2_LMS1_M6(1:400))
hold on
plot(1:400,J_c2_LMS1_M10(1:400))
hold on
title('J Learning Curves for u = 0.05*umax for case 2 as M varies')
legend('M3','M6','M10')
xlabel('n (iteration)')
ylabel('E[ | e(n) |^2 ]')

subplot(1,2,2)
plot(1:990,D_c2_LMS1_M3(1:990))
hold on
plot(1:990,D_c2_LMS1_M6(1:990))
hold on
plot(1:990,D_c2_LMS1_M10(1:990))
hold on
title('D Learning Curves for u = 0.05*umax for case 2 as M varies')
legend('M3','M6','M10')
xlabel('n (iteration)')
ylabel('E[ | curlye(n) |^2 ]')

% Comparing model orders, all seem to converge to the same value. The
% smaller the model order, the more quickly it converges, which makes sense
% because there are less taps to tune. The smaller model orders experience 
% larger swings/variance in the error.

% Notice, for case 1, it never takes more than approximately 100 iterations
% for the weights to converge. For case 2, it takes J approximately 400
% iterations to converge to the same values across models. D converges but 
% even after 1000 iterations, the three models do not agree on the same tap
% weights, with higher order M having more deviation from wiener.

%% Compare Case 1 and Case 2 = pole location
figure
subplot(1,2,1)
plot(1:250,J_c1_LMS1_M3(1:250))
hold on
plot(1:250,J_c1_LMS1_M6(1:250))
hold on
plot(1:250,J_c1_LMS1_M10(1:250))
hold on
plot(1:250,J_c2_LMS1_M3(1:250))
hold on
plot(1:250,J_c2_LMS1_M6(1:250))
hold on
plot(1:250,J_c2_LMS1_M10(1:250))
hold on
title('J Learning Curves for u = 0.05*umax as poles and M vary')
legend('C1_M3','C1_M6','C1_M10','C2_M3','C2_M6','C2_M10')
xlabel('n (iteration)')
ylabel('E[ | e(n) |^2 ]')

subplot(1,2,2)
plot(1:250,D_c1_LMS1_M3(1:250))
hold on
plot(1:250,D_c1_LMS1_M6(1:250))
hold on
plot(1:250,D_c1_LMS1_M10(1:250))
hold on
plot(1:250,D_c2_LMS1_M3(1:250))
hold on
plot(1:250,D_c2_LMS1_M6(1:250))
hold on
plot(1:250,D_c2_LMS1_M10(1:250))
hold on
title('D Learning Curves for u = 0.05*umax as poles and M vary')
legend('C1_M3','C1_M6','C1_M10','C2_M3','C2_M6','C2_M10')
xlabel('n (iteration)')
ylabel('E[ | curlye(n) |^2 ]')
% Case 1, where the poles are safely inside the stability region, converges
% much faster and has a smaller amount of error. The adaptive weights are
% closer to the Wiener filter. Case 2 begins with a MUCH larger amount of
% error and therefore takes longer to converge. Even after convergence,
% there is a larger amount of deviation. Overall, case 2 poles perform
% worse.

% LMS is stable for 0.05 and NLMS is stable. In all cases, LMS is not
% stable for 0.5 and 0.8 even though the u_max was found using the largest
% eigenvalue of R. This is because many of the LMS assumptions rely on
% small u.

% J and D behave similarly up to a scaling factor of about 4 for all.



%% 2. Adaptive Equalizer
% line up properly the d and s bc 100th d matchs to 90th s bc middle tap

% Channel Input x_n - Bernoulli sequence with x_n = +1/-1
x = randi(2,1000,1)-1;
x(x==0) = -1;

% Additive White Gaussian Noise with zero mean and variance = 0.01
v = 0.01./sqrt(2).*(randn(1000,1));

% Build an adaptive equalizer consisting of an FIR filter with 21 taps
taps_i = zeros(21,100);
taps_ii = zeros(21,100);
taps_iii = zeros(21,100);

% All of the channel transfer functions indicate that the channel preserves
% the input, delays it by one, and adds a portion of the symbol before and
% after.
% Take delta = 10, meaning taps 0-9 represent future symbols that enter
% reciever after target point and taps 11-20 represent past symbols that
% entered before target point. 
% To initialize, set the center tap to 1 and all else to 0.
taps_i(11,:) = 1;
taps_ii(11,:) = 1;
taps_iii(11,:) = 1;

% Setup adaptive algorithm based on LMS

% Dont start adaptation until the transversal filter is completely full,
% i.e. the first 21 values of u must be loaded in first

% The formula for LMS is w(n+1) = w(n) + mu*u_M(n)*conj(e(n)) where
% conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H)
    
% Perform 100 independent Monte Carlo runs to generate reasonable learning curves
J_i = zeros(100,989);
J_ii = zeros(100,989);
J_iii = zeros(100,989);

% Tune mu using the (i) channel.
% Beginning with 0.001 as recommended by Haykin, converges in approx 1000
% steps. Goal is to converge in N = 5*M to 10*M steps = 105 to 210.
% Decreasing by a factor of 10 significantly improves convergance to approx
% 600 iterations. Mu = 0.02 cuts convergance in half again. Mu = 0.05
% causes convergence to occur between 105 and 210 as required.
mu = 0.05;  

for runs = 1:100
    
    % Generate new x, u signals for each Monte Carlo experiment
    % Channel Input x_n
    x = randi(2,1002,1)-1;
    x(x==0) = -1;
    % Additive White Gaussian Noise with zero mean and variance = 0.01
    v = 0.01./sqrt(2).*(randn(1000,1));
    
    % Generate u(n) signals using the 3 different communication channel
    % transfer functions. First possible u value is u[3] if x indexing
    % begins at n=1 because each u relies on the past 2 x values.
    
    % Shift x by 1 = x[n-1]
    x_shift1 = circshift(x,1);
    % Shift x by 2 = x[n-2]
    x_shift2 = circshift(x,2);
    % Remove first 2 meaningless values
    x = x(3:end,:);
    x_shift1 = x_shift1(3:end,:);
    x_shift2 = x_shift2(3:end,:);
    
    % (i) H(z) = 0.25 + z^-1 + 0.25*z^-2 --> h[n] = 0.25*del(n) + del(n-1)
    % + 0.25*del(n-2)
    u_i = 0.25.*x + x_shift1 + 0.25.*x_shift2 + v;
    % (ii) H(z) = 0.25 + z^-1 + -0.25*z^-2 --> h[n] = 0.25*del(n) + del(n-1)
    % - 0.25*del(n-2)
    u_ii = 0.25.*x + x_shift1 - 0.25.*x_shift2 + v;
    % (iii) H(z) = -0.25 + z^-1 + 0.25*z^-2 --> h[n] = -0.25*del(n) + del(n-1)
    % + 0.25*del(n-2)
    u_iii = -0.25.*x + x_shift1 + 0.25.*x_shift2 + v;
    
    for n = 11:989
        % err = d - w^H*u where d=x(k-10)
        err_i = x(n) - taps_i(:,runs)' * flipud(u_i(n-10:n+10,1));
        err_ii = x(n) - taps_ii(:,runs)' * flipud(u_ii(n-10:n+10,1));
        err_iii = x(n) - taps_iii(:,runs)' * flipud(u_iii(n-10:n+10,1));
        
        % taps(n+1) = taps(n) + mu*u*conj(err)
        taps_i(:,runs) = taps_i(:,runs) + mu * flipud(u_i(n-10:n+10,1)) * err_i';
        taps_ii(:,runs) = taps_ii(:,runs) + mu * flipud(u_ii(n-10:n+10,1)) * err_ii';
        taps_iii(:,runs) = taps_iii(:,runs) + mu * flipud(u_iii(n-10:n+10,1)) * err_iii';
        
        % MSE Learning Curve J(n) = E(|e(n)|^2)
        J_i(runs,n-10) = abs(err_i).^2;
        J_ii(runs,n-10) = abs(err_ii).^2;
        J_iii(runs,n-10) = abs(err_iii).^2;
    end
end   

% Average J values over Monte Carlo experiments
avgJ_i = mean(J_i);
avgJ_ii = mean(J_ii);
avgJ_iii = mean(J_iii);

% Plot learning curves
figure
plot(1:250,avgJ_i(1:250))
hold on
plot(1:250,avgJ_ii(1:250))
hold on
plot(1:250,avgJ_iii(1:250))
legend('Ji','Jii','Jiii')
xlabel('n (iteration of adaptive algorithm)')
ylabel('mean squared error between output of filter and desired signal')
title('MSE Learning Curves for 3 Adaptive Equalizers')

% Time Average Estimates of R, p for u
R_i = R_erg_adapeq(u_i,21);
R_ii = R_erg_adapeq(u_ii,21);
R_iii = R_erg_adapeq(u_iii,21);

p_i = p_erg_adapeq(u_i,21);
p_ii = p_erg_adapeq(u_ii,21);
p_iii = p_erg_adapeq(u_iii,21);

% Wiener Filter: w0 = inv(R)*p
w0_i = inv(R_i)*p_i
w0_ii = inv(R_ii)*p_ii
w0_iii = inv(R_iii)*p_iii

% Jmin = minimum MSE = E(|e_0(n)|^2) =
% var(desired_signal) - w0^H * R_m * w0 = var(desired_signal) - p^H * R^-1 * p
% var(desired_signal = x) = 1
Jmin_i = 1 - w0_i' * R_i * w0_i
Jmin_ii = 1 - w0_ii' * R_ii * w0_ii
Jmin_iii = 1 - w0_iii' * R_iii * w0_iii

% Mean of Convergent Tap Weight Vector
tap_mean_i = mean(mean(taps_i))
tap_mean_ii = mean(mean(taps_ii))
tap_mean_iii = mean(mean(taps_iii))




%% 3. Adaptive Beamforming

% Import details from Problem Set 1 and 2. The same arrays will be used,
% but this time the beamformers will be adaptive.

N = 205; %number of snapshots


% PSet1 Cross Array

d_over_lambda_P1 = 0.5;

% Two linear arrays aligned across the x-axis and y-axis. Located at
% (md,0,0) and (0,md,0) where m spans from -10 to 10.
m_P1 = (-10:10).';
array_locs_P1 = zeros(3,42);
array_locs_P1(1,1:21) = m_P1; %sensors along x_axis
array_locs_P1(2,22:42) = m_P1; %sensors along y_axis
array_locs_P1(:,11) = []; %dont double count origin

% Three sources
src1_power_P1 = 1; %source to use for MVDR distortionless
src2_power_P1 = 10^(-0.5); %forced null in GSC
src3_power_P1 = 0.1; %forced null in GSC  
noise_var_P1 = 0.01;  
src_powers_P1 = [src1_power_P1; src2_power_P1; src3_power_P1];

% Angles of Arrival
thetas_P1 = [10;20;30];
phis_P1 = [20;-20;150];

% A matrix for the given parameters
[S_P1,A_P1] = steerANDdata(N,thetas_P1,phis_P1,d_over_lambda_P1,array_locs_P1,noise_var_P1,src_powers_P1);
% Theoretical Correlation Matrix R of u[n]
[R_P1,~] = cor(A_P1,S_P1,noise_var_P1,src_powers_P1);


% PSet2 Linear Array

d_over_lambda_P2 = 0.5;
M_P2 = 20; % num sensors
L_P2 = 3; % num sources
array_locs_P2 = 0:M_P2-1; %sensor array locations - uniform linear array along z axis 

% Three Sources 
src1_power_P2 = 1;
src2_power_P2 = 10^(-0.5);  
src3_power_P2 = 0.0316;     
bkgrnd_power_P2 = 0.0032; 
src_powers_P2 = [src1_power_P2; src2_power_P2; src3_power_P2];

% Angles of Arrival
thetas_P2 = [10;30;50];

% Steering Matrix
S_P2 = zeros(M_P2,L_P2);
for i=1:L_P2 
    % AOA is only a function of theta, dim: 1-by-3 for 3 sources
    a = cos(thetas_P2(i));
    % Wavenumber vector for a given planewave, dim: 1-by-3
    k = (2*pi*d_over_lambda_P2).*a;
    % Steering vector for the particular source
    S_P2(:,i) = (1/sqrt(M_P2)).*exp(-j.*k.*array_locs_P2);
end

% Data Matrix
b = (1/sqrt(2))*(randn(L_P2,N)+j*randn(L_P2,N)); %this is unit variance
b = src_powers_P2.*b; %scale by corresponding signal powers
v = (1/sqrt(2))*(randn(M_P2,N)+j*randn(M_P2,N)); %this is unit variance
v = v.*bkgrnd_power_P2; %now it has variance=noise_pwr inputed
A_P2 = S_P2*b + v;
    
% Theoretical R matrix for the environment
% R = S*R_p*S^H + noise_var*I
R_P2 = S_P2*diag(src_powers_P2)*S_P2' + bkgrnd_power_P2*eye(M_P2);


%% 
% For adaptive beamforming, the algorithm is as follows: 
% w_a(n+1) = w_a(n) + mu * Ca^H * u(n) * u^H(n) * (w_q - Ca * w_a(n)) where
% w_q = Co * (Co^H * Co) ^-1 * g
% Co is MxL constraint matrix (ex: [s(theta0) s(theta1)])
% g is Lx1 (ex: [1 0])
% (In exs, theta0 is direction of desired source and theta1 is direction of
% interfering source.)

% For MVDR, there is a single distortionless constraint, i.e. Co reduces to
% the single steering vector in the direction of interest and g=1. For GSC,
% there are three constraints, one in the distortionless direction and the
% other two are nulls.

%% Adaptive Algorithm

% Upper bound to be tuned below
mu_P1 = real(2./max(eig(R_P1))); 
mu_P2 = real(2./max(eig(R_P2)));
% Both close to 1.9 so select this
mu_3 = 1.9;

% MVDR: Single Distortionless Constraint
Co_MVDR1 = S_P1(:,1);
Co_MVDR2 = S_P2(:,1);
g_MVDR = [1];
wq_MVDR1 = Co_MVDR1*((Co_MVDR1'*Co_MVDR1)^(-1))*g_MVDR;
wq_MVDR2 = Co_MVDR2*((Co_MVDR2'*Co_MVDR2)^(-1))*g_MVDR;

% GSC: Three Constraints, force nulls
Co_GSC1 = S_P1;
Co_GSC2 = S_P2;
g_GSC = [1;0;0];
wq_GSC1 = Co_GSC1*((Co_GSC1'*Co_GSC1)^(-1))*g_GSC;
wq_GSC2 = Co_GSC2*((Co_GSC2'*Co_GSC2)^(-1))*g_GSC;

% Ca must be chosen so that Co'*Ca = 0(L x (M-L)). The columns of Ca 
% span the orthogonal complement of the columns of Co. This means that
% Ca is the orthogonal complement/null space of the range of Co.
% Use the null function to find the null space of Co and then use the
% orth function to form an orthonormal basis for the space.
Ca_MVDR1 = orth(null(Co_MVDR1'));
Ca_MVDR2 = orth(null(Co_MVDR2'));
Ca_GSC1 = orth(null(Co_GSC1'));
Ca_GSC2 = orth(null(Co_GSC2'));
% Check that full rank by calling rank(Ca) = 38 (works as required).
% Check that Co'*Ca = 0 (works as required).

% Compute w0 optimal beamformer (part B)
wa_opt_MVDR1 = inv(Ca_MVDR1'*R_P1*Ca_MVDR1)*Ca_MVDR1'*R_P1*wq_MVDR1;
wa_opt_MVDR2 = inv(Ca_MVDR2'*R_P2*Ca_MVDR2)*Ca_MVDR2'*R_P2*wq_MVDR2;
wa_opt_GSC1 = inv(Ca_GSC1'*R_P1*Ca_GSC1)*Ca_GSC1'*R_P1*wq_GSC1;
wa_opt_GSC2 = inv(Ca_GSC2'*R_P2*Ca_GSC2)*Ca_GSC2'*R_P2*wq_GSC2;

w0_opt_MVDR1 = wq_MVDR1 - Ca_MVDR1*wa_opt_MVDR1;
w0_opt_MVDR1_other_way_to_calc = (inv(R_P1)*Co_MVDR1)./(Co_MVDR1'*inv(R_P1)*Co_MVDR1); %gives same values
w0_opt_MVDR2 = wq_MVDR2 - Ca_MVDR2*wa_opt_MVDR2;
w0_opt_GSC1 = wq_GSC1 - Ca_GSC1*wa_opt_GSC1;
w0_opt_GSC2 = wq_GSC2 - Ca_GSC2*wa_opt_GSC2;


% N = num of iterations in adaptive algorithm, must correspond to number of
% snapshots in A matrix

% Matrix to hold the adjustable weight vectors, wa, for each of the 100 runs
% Size corresponds to M-L
wa_MVDR1 = zeros(40,100);
wa_MVDR12 = zeros(40,100);
wa_MVDR2 = zeros(19,100);
wa_GSC1 = zeros(38,100);
wa_GSC2 = zeros(17,100);

% Matrix to hold the final w values for each of the 100 runs
w_MVDR1 = zeros(41,100);
w_MVDR2 = zeros(20,100);
w_GSC1 = zeros(41,100);
w_GSC2 = zeros(20,100);

% Matrix to hold Mean Square Deviation Learning Curve D(n) = E(||w0-what||^2)
D_MVDR1 = zeros(100,N);
D_MVDR2 = zeros(100,N);
D_GSC1 = zeros(100,N);
D_GSC2 = zeros(100,N);

% Matrix to hold Mean Square Deviation Learning Curve J(n) = E(|e(n)|^2)
J_MVDR1 = zeros(100,N);
J_MVDR2 = zeros(100,N);
J_GSC1 = zeros(100,N);
J_GSC2 = zeros(100,N);

for runs = 1:100
    % Generate a new data matrix for each run
    % PSET1
    [~,A_P1] = steerANDdata(N,thetas_P1,phis_P1,d_over_lambda_P1,array_locs_P1,noise_var_P1,src_powers_P1);
    % PSET2
    b = (1/sqrt(2))*(randn(L_P2,N)+j*randn(L_P2,N)); 
    b = src_powers_P2.*b; 
    v = (1/sqrt(2))*(randn(M_P2,N)+j*randn(M_P2,N)); 
    v = v.*bkgrnd_power_P2;
    A_P2 = S_P2*b + v;

    % The way that A is defined in the adaptive beamforming equations is
    % that A' = [u(1) u(2) ... u(M)]. The A matrix has the u's as ROWS!
    %A_P1 = A_P1';
    %A_P2 = A_P2';
    
    rec = zeros(40,N);
    for itr = 1:N
        % w_a(n+1) = w_a(n) + mu * Ca^H * u(n) * u^H(n) * (w_q - Ca * w_a(n))
        wa_MVDR1(:,runs) = wa_MVDR1(:,runs) + 0.05*mu_3 * Ca_MVDR1' * A_P1(:,itr) * A_P1(:,itr)' * (wq_MVDR1 - Ca_MVDR1*wa_MVDR1(:,runs));
        wa_MVDR2(:,runs) = wa_MVDR2(:,runs) + 0.05*mu_3 * Ca_MVDR2' * A_P2(:,itr) * A_P2(:,itr)' * (wq_MVDR2 - Ca_MVDR2*wa_MVDR2(:,runs));
        wa_GSC1(:,runs) = wa_GSC1(:,runs) + 0.05*mu_3 * Ca_GSC1' * A_P1(:,itr) * A_P1(:,itr)' * (wq_GSC1 - Ca_GSC1*wa_GSC1(:,runs));
        wa_GSC2(:,runs) = wa_GSC2(:,runs) + 0.05*mu_3 * Ca_GSC2' * A_P2(:,itr) * A_P2(:,itr)' * (wq_GSC2 - Ca_GSC2*wa_GSC2(:,runs));
        
        % w = wq - Ca*w_a
        w_MVDR1(:,runs) = wq_MVDR1 - Ca_MVDR1*wa_MVDR1(:,runs);
        w_MVDR2(:,runs) = wq_MVDR2 - Ca_MVDR2*wa_MVDR2(:,runs);
        w_GSC1(:,runs) = wq_GSC1 - Ca_GSC1*wa_GSC1(:,runs);
        w_GSC2(:,runs) = wq_GSC2 - Ca_GSC2*wa_GSC2(:,runs);
        
        % Compute y so that J learning curve can be plotted
        % y = w^H*u
        y_MVDR1 = w_MVDR1(:,runs)' * A_P1(:,itr);
        y_MVDR2 = w_MVDR2(:,runs)' * A_P2(:,itr);
        y_GSC1 = w_GSC1(:,runs)' * A_P1(:,itr);
        y_GSC2 = w_GSC2(:,runs)' * A_P2(:,itr);
        
        y_MVDR1 = (Ca_MVDR1*wa_MVDR1(:,runs))' * A_P1(:,itr);
        y_MVDR2 = (Ca_MVDR2*wa_MVDR2(:,runs))' * A_P2(:,itr);
        y_GSC1 = (Ca_GSC1*wa_GSC1(:,runs))' * A_P1(:,itr);
        y_GSC2 = (Ca_GSC2*wa_GSC2(:,runs))' * A_P2(:,itr);
        
        % Compute d so that J learning curve can be plotted
        % d = wq^H*u
        d_MVDR1 = wq_MVDR1' * A_P1(:,itr);
        d_MVDR2 = wq_MVDR2' * A_P2(:,itr);
        d_GSC1 = wq_GSC1' * A_P1(:,itr);
        d_GSC2 = wq_GSC2' * A_P2(:,itr);
        
        % J learning curve
        J_MVDR1(runs,itr) = abs(d_MVDR1 - y_MVDR1).^2;
        J_MVDR2(runs,itr) = abs(d_MVDR2 - y_MVDR2).^2;
        J_GSC1(runs,itr) = abs(d_GSC1 - y_GSC1).^2;
        J_GSC2(runs,itr) = abs(d_GSC1 - y_GSC2).^2;
        
        % D learning curve
        D_MVDR1(runs,itr) = mean(abs(w0_opt_MVDR1-w_MVDR1(:,runs)).^2);
        D_MVDR2(runs,itr) = mean(abs(w0_opt_MVDR2-w_MVDR2(:,runs)).^2);
        D_GSC1(runs,itr) = mean(abs(w0_opt_GSC1-w_GSC1(:,runs)).^2);
        D_GSC2(runs,itr) = mean(abs(w0_opt_GSC2-w_GSC2(:,runs)).^2);
        
    end
end

% J values are all tiny, close to zero, computational errors related to the
% way numbers are stored

figure
subplot(1,2,1)
plot(1:N,mean(D_MVDR1))
hold on
plot(1:N,mean(D_MVDR2))
title('Mean Square Deviation D(n) averaged over 100 iterations for MVDR')

subplot(1,2,2)
plot(1:N,mean(D_GSC1))
hold on
plot(1:N,mean(D_GSC2))
hold on
title('Mean Square Deviation D(n) averaged over 100 iterations for GSC')

% The D curves slope upwards no matter what mu value is used. Even when mu
% is negative, the D continues to get worse over time. I have two
% theories why. One is that the algorithm is very good and so only
% numerical precision is being detected. The second is that this case is
% similar to the case discussed in lecture where a linearly increasing D is
% expected due to the way e is computed (using all past values for the
% current w vector).

%% Array Patterns
h1 = freqz(conj(w_MVDR1(:,100)));
h2 = freqz(conj(w_GSC1(:,100)));

h3 = freqz(w_MVDR2(:,100));
h4 = freqz(w_GSC2(:,100));

h5 = freqz(w0_opt_MVDR1);
h6 = freqz(w0_opt_MVDR2);

h7 = freqz(w0_opt_GSC1);
h8 = freqz(w0_opt_GSC2);

figure
subplot(1,2,1)
plot(linspace(0,180,512),20*log(abs(h1).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h2).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h5).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h7).^2))
legend('MVDR1','GSC1','MVDR1_opt','GSC1_opt')
title('MVDR and GSC Array Patterns for PSet1')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-50 50])
%xlim([0 180])

subplot(1,2,2)
plot(linspace(0,180,512),20*log(abs(h3).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h4).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h6).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h8).^2))
hold on 
legend('MVDR2','GSC2','MVDR2_opt','GSC2_opt')
title('MVDR and GSC Array Patterns for PSet2')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')
ylim([-50 80])
%xlim([0 180])


h1 = freqz(conj(w_MVDR1(:,10)));
h2 = freqz(conj(w_GSC1(:,10)));

h3 = freqz(w_MVDR2(:,10));
h4 = freqz(w_GSC2(:,10));

h5 = freqz(w0_opt_MVDR1);
h6 = freqz(w0_opt_MVDR2);

h7 = freqz(w0_opt_GSC1);
h8 = freqz(w0_opt_GSC2);


figure
sgtitle('Different w choices')
subplot(1,2,1)
plot(linspace(0,180,512),20*log(abs(h1).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h2).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h5).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h7).^2))
legend('MVDR1','GSC1','MVDR1_opt','GSC1_opt')
title('MVDR and GSC Array Patterns for PSet1')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')


subplot(1,2,2)
plot(linspace(0,180,512),20*log(abs(h3).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h4).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h6).^2))
hold on 
plot(linspace(0,180,512),20*log(abs(h8).^2))
hold on 
legend('MVDR2','GSC2','MVDR2_opt','GSC2_opt')
title('MVDR and GSC Array Patterns for PSet2')
xlabel('AOA: Theta Angle (degrees)')
ylabel('Array Pattern Value (dB)')


% The beamformers for pset1 are much noiser, less attenuation. This may
% have to do with the fact that the beamformers are really functions of
% theta and phi. However, the graph is only plotting against theta. 






%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1A. Compute R_m and p_m exactly

function [R_m,p_m] = exact_Rm_pm(p1,p2,M,beta_0)
    % To compute R_m exactly, use the formula for an AR function:
    % c * s * (-p1/(1-(p1)^2) * p1^(m) + p2/(1-(p2)^2) * p2^(m)) 
    % where s = sgn(-p1/(1+(p1)^2) + p2/(1+(p2)^2), c = (sigma_v)^2*beta and
    % beta = |-p1*(1+p2^2) + p2*(1+p1^2)|

    % Compute c and s in order to be able to calculate r[m]
    beta = abs(-p1*(1+p2.^2) + p2*(1+p1.^2));
    c = 1./beta;
    s = sign(-p1./(1+(p1.^2)) + p2./(1+(p2.^2)));

    % Anonymous function to compute r[m]
    part1 = -p1./(1-(p1.^2));
    part2 = p2./(1-(p2.^2));
    r_m = @(m) c.*s.*(part1.*(p1.^m) + part2.*(p2.^m)) 
    
    corr = r_m(0:M-1);
    R_m = toeplitz(corr);
    
    % To compute p_m exactly,
    % p is the cross correlation vector between input signal u and desired
    % signal d: p = E(uM(n)*conj(d(n))) = [r_ud(0) r_ud(-1)... r_ud(-M+1)]
    % where r_ud(-m) = E(u(n-m)*d(n)).
    % E(u[n-m]*(beta^H * u[n] + v2[n])) = E(u[n-m]*beta^H*u[n]) + E(u[n-m]*v2[n]) 
    % = E(u[n-m]*beta^H*u[n]) because v2 is uncorrelated with v1
    % and u.
    % = E(u[n-m]*beta^H*u[n]) = E(u[n-m]*(beta01*u[n] + beta02*u[n-1] +...beta06*u[n-5]))
    % = beta01*E(u[n-m]*u[n]) + beta02*E(u[n-m]*u[n-1]) +...+ beta06*E(u[n-m]*u[n-5]) =
    % = beta01*r_uu[-m] + beta02*r_uu[-m+1] + ... beta06*r_uu[-m+5] = beta01*R
    % where R is a vector of r_uu[-m+k-1].

    % Example
    % M=3
    % p_3 = [rud_0 rud_-1 rud_-2]
    % rud_-0 = beta01*r_uu[0] + beta02*r_uu[1] + ... beta06*r_uu[5]
    % rud_-1 = beta01*r_uu[-1] + beta02*r_uu[0] + ... beta06*r_uu[4]
    % rud_-2 = beta01*r_uu[-2] + beta02*r_uu[-1] + ... beta06*r_uu[3]
    
    % This can be simplified to a matrix multiply between R and beta_0
    R_10 = toeplitz(r_m(0:9));
    p_m = R_10(1:M,1:6)*beta_0'; %select applicable portion of R10
end


%% 1A. Maximum u_tilda for NLMS
function mu_tilda_max = NLMS_mu(u_n,e,D)
    % mu_max_tilda = 2*D(n)*E(|u(n)|^2)/E(|e(n)|^2) where E(|u(n)|^2) is the input
    % signal power, E(|e(n)|^2) is the error signal power and D(n) is the mean
    % squared deviation = E(||curlye(n)||^2 where curlye = w0 - what

    inpt_sgnl_pwr = mean(abs(u_n).^2);
    err_sgnl_pwr = mean(abs(e).^2); % e = d-y
    
    mu_tilda_max = (2.*inpt_sgnl_pwr.*D)./err_sgnl_pwr; 
end


%% 1B. Function to generate u and d signals
% This is necessary when running 100 iterations of LMS and NLMS. Need a new
% random signal each time.
function [u,d] = u_and_d(length,case_num)
  
    % Compute constants
    beta_0 = 1./((1:6).^2);
    % Case 1: poles are at 0.3 and 0.5
    c1_a1 = -(0.3+0.5);
    c1_a2 = 0.3*0.5;
    % Case 2: poles are at 0.3 and 0.95
    c2_a1 = -(0.3+0.95);
    c2_a2 = 0.3*0.95;

    % Unit variance, complex white noise signal v
    v1 = (1/sqrt(2))*(randn(length,1) + j*randn(length,1));
    v2 = (1/sqrt(2))*(randn(length,1) + j*randn(length,1));
    
    % Case 1
    if case_num == 1
        
        u = zeros(length,1);
        % Compute u[3]-u[6] because loop cant begin until n=7 due to constraints on
        % d[n] equation (explained below).
        u(3,:) = v1(3,:) - c1_a1*u(2,:) - c1_a2*u(1,:);
        u(4,:) = v1(4,:) - c1_a1*u(3,:) - c1_a2*u(2,:);
        u(5,:) = v1(5,:) - c1_a1*u(4,:) - c1_a2*u(3,:);
        u(6,:) = v1(6,:) - c1_a1*u(5,:) - c1_a2*u(4,:);

        d = zeros(length,1);
        % Initialize first 6 values of d1 = v2. To compute d[n], requires last 6
        % values of u because beta_0 is of length 6. Can't compute the first term
        % in d until n=7.
        d(1:6,:) = v2(1:6,:);

        % Compute u[n] = v1[n] - a1*u[n-1] - a2*u[n-2]
        for i=6:length
            u(i,:) = v1(i,:) - c1_a1*u(i-1,:) - c1_a2*u(i-2,:);
            d(i,:) = beta_0*flip(u(i-5:i,:)) + v2(i,:); %u1 must be flipped because 
            % u_m is defined as [u(n), u(n-1),... u(n-M+1)]'
        end 
    end
    
    % Case 2
    if case_num == 2
        u = zeros(length,1);
        % Compute u[3]-u[6] because loop cant begin until n=7 due to constraints on
        % d[n] equation (explained below).
        u(3,:) = v1(3,:) - c2_a1*u(2,:) - c2_a2*u(1,:);
        u(4,:) = v1(4,:) - c2_a1*u(3,:) - c2_a2*u(2,:);
        u(5,:) = v1(5,:) - c2_a1*u(4,:) - c2_a2*u(3,:);
        u(6,:) = v1(6,:) - c2_a1*u(5,:) - c2_a2*u(4,:);

        d = zeros(10^3,1);
        % Initialize first 6 values of d2 = v2. To compute d[n], requires last 6
        % values of u because beta_0 is of length 6. Can't compute the first term
        % in d until n=7.
        d(1:6,:) = v2(1:6,:);

        % Compute u[n] = v1[n] - a1*u[n-1] - a2*u[n-2]
        for i=6:length
            u(i,:) = v1(i,:) - c2_a1*u(i-1,:) - c2_a2*u(i-2,:);
            d(i,:) = beta_0*flip(u(i-5:i,:)) + v2(i,:);
        end 

    end
end
 

% Below is the code to run LMS packaged into a function
function [J,D] = LMS(mu,M,case_num,w0,numruns,len)

    % The formula for LMS is w(n+1) = w(n) + mu*u_M(n)*conj(e(n)) where
    % conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H)
    
    % Perform 100 runs to generate reasonable, non-noisy learning curves
    J = zeros(numruns,len);
    D = zeros(numruns,len);
    
    for runs = 1:numruns
        % Generate fresh u and d stochastic signals for each run
        [u,d] = u_and_d(len,case_num);
        % Initialize w
        w = zeros(M,1); 
      
        for itr = 10:len

            % y = w^H*u
            y = w'*flipud(u(itr-(M-1):itr,1)); % When pulling u values, make sure to flip vertically

            % err = d - y
            % conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H
            err = (d(itr,1) - y)';

            % w(n+1) = w(n) + mu*u_M(n)*conj(e(n))
            w = w + mu*flipud(u(itr-(M-1):itr,1))*err;

            % MSE Learning Curve J(n) = E(|e(n)|^2)
            J(runs,itr-9) = abs(d(itr,1) - y).^2;
            % Mean Square Deviation Learning Curve D(n) = E(||w0-what||^2)
            D(runs,itr-9) = norm(w0-w).^2;
            
        end
    end

    J = mean(J);
    D = mean(D);

end

%% 1B. NLMS
% Below is the code to run NLMS packaged into a function

function [J,D] = NLMS(mu,M,case_num,w0,numruns,len,lambda)

    % The formula for NLMS is w(n+1) = w(n) + mu_tilda*(1./lamd+||u_M||^2)*u_M(n)*conj(e(n))) 
    % where conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H) and lamd is greater than 0 but
    % small

    % NLMS Stepsize = 0.2*mu_tilda_max
    J = zeros(100,len); 
    D = zeros(100,len);

    % Perform 100 runs to generate reasonable, non-noisy learning curves
    for runs = 1:100
        % Generate fresh u and d for each run
        [u1,d1] = u_and_d(len,case_num);

        % Initialize w
        w = zeros(M,1);

        for itr = 10:len

            % y = w^H*u
            y = w'*flipud(u1(itr-(M-1):itr,1)); % When pulling u values, make sure to flip vertically

            % e = d - y
            e = d1(itr,1) - y;

            % mu_tilda_max
            mu = 0.2*2;

            % w(n+1) = w(n) + mu*u_M(n)*conj(e(n))
            % w(n+1) = w(n) + mu_tilda*(1./lamd+||u_M||^2)*u_M(n)*conj(e(n)))
            constants = mu./(lambda + norm(u1(itr-(M-1):itr,1)).^2);
            w = w + constants*e'*flipud(u1(itr-(M-1):itr,1));

            % MSE Learning Curve J(n) = E(|e(n)|^2)
            J(runs,itr-9) = abs(d1(itr,1) - y).^2;
            % Mean Square Deviation Learning Curve D(n) = E(||w0-what||^2)
            D(runs,itr-9) = norm(w0-w).^2;
        end
    end

    % Average over 100 runs
    J = mean(J);
    D = mean(D);
end

%% 1B. Function to graph the various J and D curves
function graphDJ(title_all,len,J1,D1,J2,D2,J3,D3,J4,D4)
   
    figure
    sgtitle(title_all)
    subplot(4,2,1)
    plot(1:len-9,J1(1:len-9))
    title('LMS J Learning Curve for u = 0.05*umax')
    xlabel('n (iteration)')
    ylabel('E[ | e(n) |^2 ]')

    subplot(4,2,2)
    plot(1:len-9,D1(1:len-9))
    title('LMS D Learning Curve for u = 0.05*umax')
    xlabel('n (iteration)')
    ylabel('E[ ||w0 - wapprox||^2 ]')

    subplot(4,2,3)
    plot(1:len-9,J2(1:len-9))
    title('LMS J Learning Curve for u = 0.5*umax')
    xlabel('n (iteration)')
    ylabel('E[ | e(n) |^2 ]')

    subplot(4,2,4)
    plot(1:len-9,D2(1:len-9))
    title('LMS D Learning Curve for u = 0.5*umax')
    xlabel('n (iteration)')
    ylabel('E[ ||w0 - wapprox||^2 ]')

    subplot(4,2,5)
    plot(1:len-9,J3(1:len-9))
    title('LMS J Learning Curve for u = 0.5*umax')
    xlabel('n (iteration)')
    ylabel('E[ | e(n) |^2 ]')

    subplot(4,2,6)
    plot(1:len-9,D3(1:len-9))
    title('LMS D Learning Curve for u = 0.5*umax')
    xlabel('n (iteration)')
    ylabel('E[ ||w0 - wapprox||^2 ]')
    
    subplot(4,2,7)
    plot(1:len-9,J4(1:len-9))
    title('NLMS J Learning Curve for u = 0.2*umax')
    xlabel('n (iteration)')
    ylabel('E[ | e(n) |^2 ]')

    subplot(4,2,8)
    plot(1:len-9,D4(1:len-9))
    title('NLMS D Learning Curve for u = 0.2*umax')
    xlabel('n (iteration)')
    ylabel('E[ ||w0 - wapprox||^2 ]')

end

%% (2) Function to estimate R ergodically
function R = R_erg_adapeq(x,M)
    
    % Store the r[m] values
    r = zeros(1,M);
    
    for lag=0:M-1
        % Chop off M last values of x(n) because the corresponding x(n+m) are
        % not available
        x_n = x(1:end-lag);
        % Remove first M values of x(n+m) because the corresponding x(n) values
        % are not available
        x_nm = x(lag+1:end); %because MATLAB indexes from 1, must add 1 to M

        r(1,lag+1) = mean(x_n.*x_nm);
    end
    
    R = toeplitz(r);
end

%% (2) Function to estimate p ergodically
function p = p_erg_adapeq(x,M)
    
    % Store the r[m] values
    p = zeros(M,1);
    
    for lag=1:M
        % Chop off M last values of x(n) because the corresponding x(n+m) are
        % not available
        x_n = x(1:end-lag);
        % Remove first M values of x(n+m) because the corresponding x(n) values
        % are not available
        x_nm = x(lag+1:end); %because MATLAB indexes from 1, must add 1 to M

        p(lag,1) = mean(x_n.*x_nm);
    end
end
%% Functions imported from PSet1 for use in problem 3

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

function [R_theor,R_erg] = cor(A,S,noise_pwr,source_pwr)
    [M,N] = size(A); %num of sensors and time samples
    [~,L] = size(S); %num of sources

    % Theoretical
    %R = S*R_p*S^H + noise_var*I
    R_theor = S*diag(source_pwr)*S' + noise_pwr*eye(M);
    
    % Approximate/Ergodic Correlation - outer product of data A matrix
    R_erg = (1/N).*(A*A'); %A*A^H %add 1/k portion
end

