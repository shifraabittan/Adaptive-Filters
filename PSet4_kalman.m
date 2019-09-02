%% Shifra Abittan
% Prof Fontaine
% ECE416 Adaptive Filters
% PSet4 Kalman Filter Project

% This project implements and studies the performance of several forms of 
% the discrete-time Kalman filter, including: 
% (1) standard/covariance (2) information (3) square-root covariance (4)
% extended (EKF)

%% System Setup

% All data is real. If u is a vector and A is a matrix, then |u| = Euclidean 
% length, ||A|| = spectral norm

% System model is time-varying and given by:
%           x(n+1) = A(n)*x(n) + v(n)
%           y(n) = C(n)*x(n) + w(n)

% For iterations 1<=n<=N1, A(n) = A1 and for iterations N1+1<=n<=N1+N2,
% A(n) = A2
A1 = [-0.9 1 0 0; 0 -0.9 0 0; 0 0 -0.5 0.5; 1 0 -0.5 -0.5];
A2 = [0.9 1 0 0; 0 0.9 0 0; 0 0 -0.5 0.5; 1 0 -0.5 -0.5];

C = [1 -1 1 -1; 0 1 0 1];

% The covariance matrices Qv and Qw of the process noise v(n) and
% observation noise w(n) are constant and given as:
Qv = [0.25 0.25 0 0; 0.25 0.5 0.25 0; 0 0.25 0.5 0.25; 0 0 0.25 0.5];
Qw = 0.1*eye(2);


%% Studying the Prescribed System

% Compute eigenvalues of A1 and A2
[evec_A1,eig_A1] = eig(A1)
[evec_A2,eig_A2] = eig(A2)

% Confirm: 

% 1. A1 and A2 are stable --> if magnitude of all eigenvalues is smaller
% than 1, then matrix is stable
mag_eig_A1 = abs(eig_A1);
if (mag_eig_A1<[1;1;1;1])
    disp('A1 is stable')
end

mag_eig_A2 = abs(eig_A2);
if (mag_eig_A1<[1;1;1;1])
    disp('A2 is stable')
end

% 2. Each has an eigenvalue of algebraic multiplicity 2 but geometric
% multiplicity 1
% The eigenvalue -0.9 for A1 and 0.9 for A2 has algebraic multiplicity 2.
% To check the geometric multiplicity: find the dimension of the nullspace
% of A-eig*I = eigenspace of A for the particular eig. Alternatively, looking 
% at the eigenvectors shows that the 2 vectors corresponding to the
% -0.9/0.9 eigenvalue are linearly dependent --> the geometric multiplicity
% is 1.
linind_A1 = rank(evec_A1); %rank = 3 (dim = 4)
linind_A2 = rank(evec_A1); %rank = 3 (dim = 4)

% 3. Each has a pair of complex eigenvalues --> (-0.5 + 0.5j),(-0.5 - 0.5j)
% in each

% In A1, all the eigenvalues lie in the lefthand side of the plane. In A2,
% the 0.9 eigenvalue is in the right hand plane/reflected over y axis.

% Given the following model, x(n+1) = A*x(n) + v(n) where v(n) is white
% with covariance matrix Qv and stable A. The steady state covariance
% matrix of x(n) satisfies the discrete-time Lyapunov equation: 
% K = AKA^T + Qv. Find the solution to the discrete-time Lyapunov equation
% for A1, A2 to understand the size of the x vector at steady-state.
K_A1 = dlyap(A1,Qv)
K_A2 = dlyap(A2,Qv)

% The eigenvalues of A give a sense of how long until steady state
% condition is met. p is the magnitude of the dominant eigenvalue in A and
% N is the smallest values such that p^N < 0.01. N models the time until
% the effect of the initial condition x(0) dissipates.
p = 0.9; %for both A1 and A2
N = 44; %p^N = 0.0097 < 0.01
M=N*2;
% Check that (A,C) is observable by forming the observability matrix and
% computing its rank.
% O(A,C) = [C; CA; CA^2; ... CA^N-1]. If the O(A,C) matrix, which is Np x N
% has full rank N, then (A,C) is observable. 
% In this case, N=4 and p=2 (dim of C is 2x4)
obs_A1 = [C; C*A1; C*(A1^2); C*(A1^3)]; 
rank_obs_A1 = rank(obs_A1)
if rank_obs_A1 == 4
    disp('(A1,C) is observable')
end

obs_A2 = [C; C*A2; C*(A2^2); C*(A2^3)]; 
rank_obs_A2 = rank(obs_A2)
if rank_obs_A2 == 4
    disp('(A2,C) is observable')
end

% Because (A1,C) and (A2,C) are observable and Qv is full rank, there is a
% steady-state prediction error covariance matrix that solves an algebraic
% Riccati equation (ARE).

% The MATLAB function dare solves 
% E'XE = A'XA - (A'XB + S)(B'XB + R)^-1 *(A'XB + S)' + Q
% In the notes, the problem is posed as 
% K = AKA^H + Qv - AKC^H(CKC^H + Qw)^-1* CKA^H.
% Therefore, E = I (default)
%            X = K
%            A = A^H
%            B = C^H
%            S = zeros (default)
%            R = Qw
%            Q = Qv
[K1_ss,~,~] = dare(A1',C',Qv,Qw)
[K2_ss,~,~] = dare(A2',C',Qv,Qw)

% Confirm that K1_ss and K2_ss are positive definite by checking the
% eigenvalues
eig_K1_ss = eig(K1_ss)
eig_K2_ss = eig(K2_ss)
% All eigenvalues are positive = K_ss is pd

% Compute the spectral norm ||K1_ss - K2_ss||
spec_norm_btwn_Kss = norm(K1_ss - K2_ss)

% Experiment 

x = zeros(4,M);
% Initial Condition
x(:,1) = [1;1;1;1]; 

% Generate white noise with covariance matrix Qv
M = N*2; %N=44 computed above
unit_var_white = 1/sqrt(2)*(randn(4,M)); % white noise with unit variance
v = (Qv^0.5)*unit_var_white; % white noise with covariance matrix Qv

% Run the process equation x(n+1) = A*x(n) + v(n) for N time steps using A1
for n = 2:N+1
    x(:,n) = A1*x(:,n-1) + v(:,n);
end
% Run the process equation x(n+1) = A*x(n) + v(n) for N time steps using A2
for n = N+2:M
    x(:,n) = A2*x(:,n-1) + v(:,n);
end

% Plot state vector vs time
figure
plot(1:M,x)
hold on
xline(44.5,'--')
title('State Vector x Versus Time');
xlabel('time[n]')
ylabel('State Vector Values')
legend('1st component of x','2nd component of x','3rd component of x','4th component of x','Switch from A1 to A2')

%% Covariance Kalman Filter - core function to run one iteration of the filter

% Core function that runs one iteration of the Kalman filter. 
% Inputs are x_h(n|n-1) [refer to as x_in], 
%            K(n,n-1) [refer to as K_in]
% Outputs are x_h(n|n) [refer to as x_filt], 
%             x_h(n+1|n) [refer to as x_pred], 
%             K(n,n), 
%             K(n+1,n) [refer to as K_futn_n], 
%             G(n)

%{
function [x_filt,x_pred,K_n_n,K_futn_n,G] = core_kalman_covar(x_in,K_in,y,A,C,Qw,Qv)
    % Compute Kalman gain = K(n,n-1)*C^H*(C*K(n,n-1)*C^H + Qw)^-1
    G = K_in*C'*inv(C*K_in*C' + Qw);
    
    % Compute Kalman innovations = y(n) - C*x(n|n-1) =  white noise by orthogonality
    alpha = y - C*x_in;
    
    % Compute filtering x = x_h(n|n) = x_h(n|n-1) + G*alpha
    x_filt = x_in + G*alpha;
    % Compute prediction x = x_h(n+1|n) = A*x_h(n|n)
    x_pred = A*x_filt;
    
    % Update K values for next iteration
    K_n_n = K_in - G*C*K_in;
    K_futn_n = A*K_n_n*A' + Qv;
end
%}

%% Preliminary Study

% Perform 100 Monte Carlo simulations of the Kalman Filter with A1/constant
% and determine the average number of iterations it takes for 
% ||K(n,n-1) - K1,ss||/||K1_ss|| < 0.01.

% Variables to store from each simulation:
Knn_neg_count = []; %Count how many negative values appear in K_n_n
iter_count = zeros(100,1); %Record num iter until convergance

for z = 1:100
    
    % Initial x(0) (actual) is a vector with iid N(0,1) components
    x_n_actual = randn(4,1);
    % Initialize x_h[1|0] = 0
    x_n_nm1 = zeros(4,1);
    % Initialize K(1,0) to identity matrix
    K_n_nm1 = eye(4);
    K_n_nm1_rec = zeros(100,20);

    while (norm(K_n_nm1 - K1_ss)./norm(K1_ss)) > 0.01
        
        % Keep track of required num of KF iterations
        iter_count(z,1) = iter_count(z,1) + 1;

        % In order to implement Kalman filter, which approximates the state vector 
        % x from the y output measurements, we must generate the y values. Generation
        % of the y values requires knowledge of the actual x values as well. The
        % equations governing actual x and y are: 
        % x(n+1) = A*x(n) + v(n)
        % y(n) = C*x(n) + w(n)

        % Generate v(n)
        unit_white = 1/sqrt(2)*(randn(4,1)); % white noise with unit variance
        v = (Qv^0.5)*unit_white; % white noise with covariance matrix Qv
        % Generate x(n+1)
        x_n_actual = A1*x_n_actual + v; %update x_n_actual
        % for first iteration, x(0) is given and the above computes x(1). This
        % is necessary because y sequence begins at y(1)

        % Generate w(n)
        unit_white = 1/sqrt(2)*(randn(2,1)); % white noise with unit variance
        w = (Qw^0.5)*unit_white; % white noise with covariance matrix Qw
        % Generate y(n)
        y = C*x_n_actual + w; %y(n) = C*x(n) + w(n)

        % Now run one iteration of the kalman filter to approximate x(n|n) and
        % x(n+1|n)
        [x_n_n,x_np1_n,K_n_n,K_np1_n,G] = core_kalman_covar(x_n_nm1,K_n_nm1,y,A1,C,Qw,Qv);

        % Update values
        x_n_nm1 = x_np1_n; % x_h[n|n-1] = x_h[n+1|n], e.g. x[2|1]_nextitr = x[1+1|1]_previtr_outputKF
        K_n_nm1 = K_np1_n; % K(n,n-1) = K(n+1,n), e.g. K(2,1)_nextitr = K(1+1,1)_previtr_outputKF

        % Check if K(n,n) is strictly positive
        num_neg = sum(sum(K_n_n<0)); %compute how many values are negative
        Knn_neg_count = [Knn_neg_count; iter_count(z,1) num_neg];

    end
end

num_iter_A1 = mean(iter_count)

% Interestingly, each of the 100 iterations performs almost identically.

% Repeat for A2.
% Variables to store from each simulation:
Knn_neg_count_A2 = []; %Count how many negative values appear in K_n_n
iter_count_A2 = zeros(100,1); %Record num iter until convergance

for z = 1:100
    
    % Initial x(0) (actual) is a vector with iid N(0,1) components
    x_n_actual = randn(4,1);
    % Initialize x_h[1|0] = 0
    x_n_nm1 = zeros(4,1);
    % Initialize K(1,0) to identity matrix
    K_n_nm1 = eye(4);
    K_n_nm1_rec = zeros(100,20);

    while (norm(K_n_nm1 - K2_ss)./norm(K2_ss)) > 0.01
        
        % Keep track of required num of KF iterations
        iter_count_A2(z,1) = iter_count_A2(z,1) + 1;

        % In order to implement Kalman filter, which approximates the state vector 
        % x from the y output measurements, we must generate the y values. Generation
        % of the y values requires knowledge of the actual x values as well. The
        % equations governing actual x and y are: 
        % x(n+1) = A*x(n) + v(n)
        % y(n) = C*x(n) + w(n)

        % Generate v(n)
        unit_white = 1/sqrt(2)*(randn(4,1)); % white noise with unit variance
        v = (Qv^0.5)*unit_white; % white noise with covariance matrix Qv
        % Generate x(n+1)
        x_n_actual = A2*x_n_actual + v; %update x_n_actual
        % for first iteration, x(0) is given and the above computes x(1). This
        % is necessary because y sequence begins at y(1)

        % Generate w(n)
        unit_white = 1/sqrt(2)*(randn(2,1)); % white noise with unit variance
        w = (Qw^0.5)*unit_white; % white noise with covariance matrix Qw
        % Generate y(n)
        y = C*x_n_actual + w; %y(n) = C*x(n) + w(n)

        % Now run one iteration of the kalman filter to approximate x(n|n) and
        % x(n+1|n)
        [x_n_n,x_np1_n,K_n_n,K_np1_n,G] = core_kalman_covar(x_n_nm1,K_n_nm1,y,A2,C,Qw,Qv);

        % Update values
        x_n_nm1 = x_np1_n; % x_h[n|n-1] = x_h[n+1|n], e.g. x[2|1]_nextitr = x[1+1|1]_previtr_outputKF
        K_n_nm1 = K_np1_n; % K(n,n-1) = K(n+1,n), e.g. K(2,1)_nextitr = K(1+1,1)_previtr_outputKF

        % Check if K(n,n) is strictly positive
        num_neg = sum(sum(K_n_n<0)); %compute how many values are negative
        Knn_neg_count_A2 = [Knn_neg_count_A2; iter_count(z,1) num_neg];

    end
end

num_iter_A2 = mean(iter_count_A2)

% Both N1 and N2 = average number of iterations for convergence are less
% than 10. Therefore, replace both as 10.

%% Run the routine where the first N1 iterations uses A1 and the next N2 
% iterations uses A2.

% Store variables
x_actual = zeros(4,21); %first value is x(0)
x_n_nm1 = zeros(4,21); % Initialize x_h[1|0] = 0
x_n_n = zeros(4,21); %first value is x[0|0]
y = zeros(2,20); %first value is y(1)

K_n_n = zeros(4,4*20);

norm_Khat_Kss = zeros(1,20);
norm_K_n_nm1 = zeros(1,20);
norm_Knn = zeros(1,20);

% Initial Conditions
% x(0) = vector with iid N(0,1) components
x_actual(:,1) = randn(4,1); 
% Initialize K(1,0) to identity matrix
K_n_nm1 = eye(4);

% Generate v
unit_var_white = 1/sqrt(2)*(randn(4,20));
v = (Qv^0.5)*unit_var_white; 
% Generate w(n)
unit_white = 1/sqrt(2)*(randn(2,20)); 
w = (Qw^0.5)*unit_white; 

for n = 2:11
    % Process equation x(n+1) = A*x(n) + v(n) for N1 time steps using A1
    x_actual(:,n) = A1*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % Compute ||K(n,n-1) - K1_ss||
    norm_Khat_Kss(1,n-1) = norm(K_n_nm1 - K1_ss);
    % Compute ||K(n,n-1)||
    norm_K_n_nm1(1,n-1) = norm(K_n_nm1);
    
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [x_n_n(:,n-1),x_n_nm1(:,n),K_n_n(:,((n-1)*4)-3:(n-1)*4),K_np1_n,G] = core_kalman_covar(x_n_nm1(:,n-1),K_n_nm1,y(:,n-1),A1,C,Qw,Qv);

    % Compute ||K(n,n)||
    norm_Knn(1,n-1) = norm(K_n_n(:,((n-1)*4)-3:(n-1)*4));
   
    % Update values
    K_n_nm1 = K_np1_n;
end

for n = 12:21
    % Process equation x(n+1) = A*x(n) + v(n) for N2 time steps using A2
    x_actual(:,n) = A2*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % Compute ||K(n,n-1) - K1_ss||
    norm_Khat_Kss(1,n-1) = norm(K_n_nm1 - K2_ss);
    % Compute ||K(n,n-1)||
    norm_K_n_nm1(1,n-1) = norm(K_n_nm1);
    
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [x_n_n(:,n-1),x_n_nm1(:,n),K_n_n(:,((n-1)*4)-3:(n-1)*4),K_np1_n,G] = core_kalman_covar(x_n_nm1(:,n-1),K_n_nm1,y(:,n-1),A2,C,Qw,Qv);

    % Compute ||K(n,n)||
    norm_Knn(1,n-1) = norm(K_n_n(:,((n-1)*4)-3:(n-1)*4));
   
    % Update values
    K_n_nm1 = K_np1_n;
end

% Plot results

% Graph |x(n)|
figure
subplot(2,2,1)
plot(0:10,vecnorm(x_actual(:,1:11)),'r.-')
hold on
plot(10:20,vecnorm(x_actual(:,11:21)),'b.-')
hold on
xline(10,'--')
title('|x(n)| = Euclidean Length of Actual State Variables')
xlabel('time[n]')
legend('A1','A2')

% Generally A1 causes norm to increase and A2 causes the norm to decrease
% 3-D plot of (x1,x2,x3) to visualize actual state trajectory, forming a
% shape that somewhat resembles a bumpy Gaussian distribution shape

subplot(2,2,2)
plot3(x_actual(1,1:11),x_actual(2,1:11),x_actual(3,1:11),'r.-')
hold on
plot3(x_actual(1,11:21),x_actual(2,11:21),x_actual(3,11:21),'b.-')
legend('A1','A2')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('Plot of (x1,x2,x3) components of the actual state vector to visualize trajectory')

% By running section multiple times, can see multiple trial runs

% Graph of | x(n) - x_h(n|n-1) | and | x(n) - x_h(n|n) | where n ranges
% from n=1 to n=20
norm_xpast = vecnorm(x_actual(:,2:21) - x_n_nm1(:,2:21));
norm_xpastandpres = vecnorm(x_actual(:,2:21) - x_n_n(:,1:20));

subplot(2,2,3:4)
plot(1:20,norm_xpast,'m')
hold on
plot(1:20,norm_xpastandpres,'g')
title('Error of Kalman Prediction and Kalman Filtering over time')
xlabel('time[n]')
ylabel('Euclidean Length of the Error')
legend('Kalman Prediction | x(n) - x_h(n|n-1) |','Kalman Filtering | x(n) - x_h(n|n) |')

% Notice the tremendous improvement/correction achieved by filtering.

figure
subplot(2,1,1)
plot(1:10,norm_Khat_Kss(1:10),'r.-')
hold on
plot(11:20,norm_Khat_Kss(11:20),'b.-')
title('Error between K(n,n-1) generated from Kalman filter and Ki,ss where i = 1 for N=1:10 and i = 2 for N=11:20')
xlabel('time[n]')

% Validates choice of N1=5, N2=4 above

% Compute ||K(n,n) - K(n-1,n-1)||
% This sequence begins at K(2,2)-K(1,1) because K(0,0) doesnt exist
Knn_start2 = K_n_n(:,5:end); %remove K(1,1)
Knn_start1 = K_n_n(:,1:end-4); %remove K(20,20)
K_minus_Kprev = Knn_start2 - Knn_start1;
norm_K_Kprev = zeros(1,19);
for l=1:19
   norm_K_Kprev(1,l) = norm(K_minus_Kprev(:,(l*4)-3:l*4));
end

% Graph of ||K(n,n-1)||, ||K(n,n)||, ||K(n,n) - K(n-1,n-1)||
subplot(2,1,2)
plot(1:20,norm_K_n_nm1,'.-')
hold on
plot(1:20,norm_Knn,'.-')
hold on
plot(2:20,norm_K_Kprev,'.-')
legend('|| K(n,n-1) ||','|| K(n,n) ||','|| K(n,n) - K(n-1,n-1) ||')
xlabel('n')
title('Various Norms across iterations: ||K(n,n-1)||, ||K(n,n)||, ||K(n,n) - K(n-1,n-1)||')








%% Information Kalman Filter

%% Information Kalman Filter - core function to run one iteration of the filter

% Core function that runs one iteration of the information kalman filter. 
% Inputs are chi(n,n-1) [refer to as chi_in], 
%            P(n,n-1) [refer to as P_in]
% Outputs are chi(n,n) [refer to as chi_filt], 
%             chi(n+1,n) [refer to as chi_pred], 
%             P(n,n) [refer to as Pnn], 
%             P(n+1,n) [refer to as P_futn_n], 

%{
function [chi_filt,chi_pred,Pnn,P_futn_n] = core_kalman_info(chi_in,P_in,y,A,C,Qw,Qv)
    % P(n,n) = P(n,n-1) + C^H*Qw^-1*C
    Pnn = P_in + C'*inv(Qw)*C;

    % M(n) = A^(-H)*P(n,n)*A^(-1)
    M = inv(A)'*Pnn*inv(A);

    % F(n) = (I + M*Qv)^-1
    F = inv(eye(4) + M*Qv);

    % P(n+1,n) = F*M
    P_futn_n = F*M;
    
    % chi_filt = chi(n,n) = chi(n,n-1) + C^H*Qw^-1*y(n)
    chi_filt = chi_in + C'*inv(Qw)*y;
    % chi_pred = chi(n+1,n) = F*A^-H*chi_filt
    chi_pred = F*inv(A)'*chi_filt;
end
%}

%% Preliminary Study

% Perform 100 Monte Carlo simulations of the Information Kalman Filter with
% A1/constant and determine the average number of iterations it takes for 
% ||P(n,n-1) - inv(K1,ss)||/||inv(K1_ss)|| < 0.01.

% Variables to store from each simulation:
Pnn_neg_count = []; %Count how many negative values appear in P_n_n
iter_count = zeros(100,1); %Record num iter until convergance

for z = 1:100
    
    % Initial x(0) (actual) is a vector with iid N(0,1) components
    x_n_actual = randn(4,1);
    % Initialize x_h[1|0] = 0
    chi_n_nm1 = zeros(4,1);
    % Initialize P(1,0) to identity matrix
    P_n_nm1 = eye(4);
    P_n_nm1_rec = zeros(100,20);

    while (norm(P_n_nm1 - inv(K1_ss))./norm(inv(K1_ss))) > 0.01
        
        % Keep track of required num of IKF iterations
        iter_count(z,1) = iter_count(z,1) + 1;

        % In order to implement IKF, which approximates the state vector 
        % x from the y output measurements, we must generate the y values. Generation
        % of the y values requires knowledge of the actual x values as well. The
        % equations governing actual x and y are: 
        % x(n+1) = A*x(n) + v(n)
        % y(n) = C*x(n) + w(n)

        % Generate v(n)
        unit_white = 1/sqrt(2)*(randn(4,1)); % white noise with unit variance
        v = (Qv^0.5)*unit_white; % white noise with covariance matrix Qv
        % Generate x(n+1)
        x_n_actual = A1*x_n_actual + v; %update x_n_actual
        % for first iteration, x(0) is given and the above computes x(1). This
        % is necessary because y sequence begins at y(1)

        % Generate w(n)
        unit_white = 1/sqrt(2)*(randn(2,1)); % white noise with unit variance
        w = (Qw^0.5)*unit_white; % white noise with covariance matrix Qw
        % Generate y(n)
        y = C*x_n_actual + w; %y(n) = C*x(n) + w(n)

        % Now run one iteration of the kalman filter to approximate chi(n|n) and
        % chi(n+1|n)
        [chi_n_n,chi_np1_n,P_n_n,P_np1_n] = core_kalman_info(chi_n_nm1,P_n_nm1,y,A1,C,Qw,Qv);
        
        % Update values
        chi_n_nm1 = chi_np1_n; 
        P_n_nm1 = P_np1_n; 

        % Check if P(n,n) is strictly positive
        num_neg = sum(sum(P_n_n<0)); %compute how many values are negative
        Pnn_neg_count = [Pnn_neg_count; iter_count(z,1) num_neg];

    end
end

num_iter_A1_IKF = mean(iter_count)

% Repeat for A2.
% Variables to store from each simulation:
Pnn_neg_count_A2 = []; %Count how many negative values appear in P_n_n
iter_count_A2 = zeros(100,1); %Record num iter until convergance

for z = 1:100
    
    % Initial x(0) (actual) is a vector with iid N(0,1) components
    x_n_actual = randn(4,1);
    % Initialize x_h[1|0] = 0
    chi_n_nm1 = zeros(4,1);
    % Initialize K(1,0) to identity matrix
    P_n_nm1 = eye(4);
    P_n_nm1_rec = zeros(100,20);

    while (norm(P_n_nm1 - inv(K2_ss))./norm(inv(K2_ss))) > 0.01
        
        % Keep track of required num of KF iterations
        iter_count_A2(z,1) = iter_count_A2(z,1) + 1;

        % In order to implement Kalman filter, which approximates the state vector 
        % x from the y output measurements, we must generate the y values. Generation
        % of the y values requires knowledge of the actual x values as well. The
        % equations governing actual x and y are: 
        % x(n+1) = A*x(n) + v(n)
        % y(n) = C*x(n) + w(n)

        % Generate v(n)
        unit_white = 1/sqrt(2)*(randn(4,1)); % white noise with unit variance
        v = (Qv^0.5)*unit_white; % white noise with covariance matrix Qv
        % Generate x(n+1)
        x_n_actual = A2*x_n_actual + v; %update x_n_actual
        % for first iteration, x(0) is given and the above computes x(1). This
        % is necessary because y sequence begins at y(1)

        % Generate w(n)
        unit_white = 1/sqrt(2)*(randn(2,1)); % white noise with unit variance
        w = (Qw^0.5)*unit_white; % white noise with covariance matrix Qw
        % Generate y(n)
        y = C*x_n_actual + w; %y(n) = C*x(n) + w(n)

        % Now run one iteration of the kalman filter to approximate chi(n|n) and
        % chi(n+1|n)
        [chi_n_n,chi_np1_n,P_n_n,P_np1_n] = core_kalman_info(chi_n_nm1,P_n_nm1,y,A2,C,Qw,Qv);

        % Update values
        chi_n_nm1 = chi_np1_n; 
        P_n_nm1 = P_np1_n;

        % Check if P(n,n) is strictly positive
        num_neg = sum(sum(P_n_n<0)); %compute how many values are negative
        Pnn_neg_count_A2 = [Pnn_neg_count_A2; iter_count(z,1) num_neg];

    end
end

num_iter_A2_IKF = mean(iter_count_A2)

% Both N1 and N2 were reduced by 1 when using IKF over Covariance Kalman
% Filter.
%   Standard KF: N1 = 5; N2 = 4
%   Information KF: N1 = 4; N2 = 3


%% Run the routine where the first N1 iterations uses A1 and the next N2 
% iterations uses A2.

% Store variables
x_actual = zeros(4,21); %first value is x(0)
chi_n_nm1 = zeros(4,21); % Initialize chi_h[1|0] = 0
chi_n_n = zeros(4,21); %first value is chi[0|0]
y = zeros(2,20); %first value is y(1)

P_n_n = zeros(4,4*20);

norm_Phat_Pss = zeros(1,20);
norm_P_n_nm1 = zeros(1,20);
norm_Pnn = zeros(1,20);

% Initial Conditions
% x(0) = vector with iid N(0,1) components
x_actual(:,1) = randn(4,1); 
% Initialize K(1,0) to identity matrix
P_n_nm1 = eye(4);

% Generate v
unit_var_white = 1/sqrt(2)*(randn(4,20));
v = (Qv^0.5)*unit_var_white; 
% Generate w(n)
unit_white = 1/sqrt(2)*(randn(2,20)); 
w = (Qw^0.5)*unit_white; 

for n = 2:11
    % Process equation x(n+1) = A*x(n) + v(n) for N1 time steps using A1
    x_actual(:,n) = A1*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % Compute ||P(n,n-1) - inv(K1_ss)||
    norm_Phat_Pss(1,n-1) = norm(P_n_nm1 - inv(K1_ss));
    % Compute ||P(n,n-1)||
    norm_P_n_nm1(1,n-1) = norm(P_n_nm1);
    
    % Run one iteration Kalman filter to approximate chi(n|n) and chi(n+1|n)
    [chi_n_n(:,n-1),chi_n_nm1(:,n),P_n_n(:,((n-1)*4)-3:(n-1)*4),P_np1_n] = core_kalman_covar(chi_n_nm1(:,n-1),P_n_nm1,y(:,n-1),A1,C,Qw,Qv);

    % Compute ||P(n,n)||
    norm_Pnn(1,n-1) = norm(P_n_n(:,((n-1)*4)-3:(n-1)*4));
   
    % Update values
    P_n_nm1 = P_np1_n;
end

for n = 12:21
    % Process equation x(n+1) = A*x(n) + v(n) for N2 time steps using A2
    x_actual(:,n) = A2*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % Compute ||P(n,n-1) - inv(K2_ss)||
    norm_Phat_Pss(1,n-1) = norm(P_n_nm1 - inv(K2_ss));
    % Compute ||P(n,n-1)||
    norm_P_n_nm1(1,n-1) = norm(P_n_nm1);
    
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [chi_n_n(:,n-1),chi_n_nm1(:,n),P_n_n(:,((n-1)*4)-3:(n-1)*4),P_np1_n] = core_kalman_covar(chi_n_nm1(:,n-1),P_n_nm1,y(:,n-1),A2,C,Qw,Qv);

    % Compute ||P(n,n)||
    norm_Pnn(1,n-1) = norm(P_n_n(:,((n-1)*4)-3:(n-1)*4));
   
    % Update values
    P_n_nm1 = P_np1_n;
end

% Plot results

% Graph |x(n)|
figure
subplot(2,2,1)
plot(0:10,vecnorm(x_actual(:,1:11)),'r.-')
hold on
plot(10:20,vecnorm(x_actual(:,11:21)),'b.-')
hold on
xline(10,'--')
title('IKF |x(n)| = Euclidean Length of Actual State Variables')
xlabel('time[n]')
legend('A1','A2')

% Generally A1 causes norm to increase and A2 causes the norm to decrease
% 3-D plot of (x1,x2,x3) to visualize actual state trajectory, forming a
% shape that somewhat resembles a bumpy Gaussian distribution shape

subplot(2,2,2)
plot3(x_actual(1,1:11),x_actual(2,1:11),x_actual(3,1:11),'r.-')
hold on
plot3(x_actual(1,11:21),x_actual(2,11:21),x_actual(3,11:21),'b.-')
legend('A1','A2')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('IKF Plot of (x1,x2,x3) components of the actual state vector to visualize trajectory')

% By running section multiple times, can see multiple trial runs

% Graph of | x(n) - chi(n|n-1) | and | x(n) - chi(n|n) | where n ranges
% from n=1 to n=20
norm_chi_past = vecnorm(x_actual(:,2:21) - chi_n_nm1(:,2:21));
norm_chi_pastandpres = vecnorm(x_actual(:,2:21) - chi_n_n(:,1:20));

subplot(2,2,3:4)
plot(1:20,norm_chi_past,'m')
hold on
plot(1:20,norm_chi_pastandpres,'g')
title('Error of Information Kalman Prediction and Information Kalman Filtering over time')
xlabel('time[n]')
ylabel('Euclidean Length of the Error')
legend('Kalman Prediction | x(n) - x_h(n|n-1) |','Kalman Filtering | x(n) - x_h(n|n) |')

% Notice the tremendous improvement/correction achieved by filtering.

figure
subplot(2,1,1)
plot(1:10,norm_Phat_Pss(1:10),'r.-')
hold on
plot(11:20,norm_Phat_Pss(11:20),'b.-')
title('Error between P(n,n-1) generated from IKF and inv(Ki,ss) where i = 1 for N=1:10 and i = 2 for N=11:20')
xlabel('time[n]')

% Validates choice of N1=5, N2=4 above

% Compute ||P(n,n) - P(n-1,n-1)||
% This sequence begins at P(2,2)-P(1,1) because P(0,0) doesnt exist
Pnn_start2 = P_n_n(:,5:end); %remove P(1,1)
Pnn_start1 = P_n_n(:,1:end-4); %remove P(20,20)
P_minus_Pprev = Pnn_start2 - Pnn_start1;
norm_P_Pprev = zeros(1,19);
for l=1:19
   norm_P_Pprev(1,l) = norm(P_minus_Pprev(:,(l*4)-3:l*4));
end

% Graph of ||P(n,n-1)||, ||P(n,n)||, ||P(n,n) - P(n-1,n-1)||
subplot(2,1,2)
plot(1:20,norm_P_n_nm1,'.-')
hold on
plot(1:20,norm_Pnn,'.-')
hold on
plot(2:20,norm_P_Pprev,'.-')
legend('|| P(n,n-1) ||','|| P(n,n) ||','|| P(n,n) - P(n-1,n-1) ||')
xlabel('n')
title('IKF Various Norms across iterations: ||P(n,n-1)||, ||P(n,n)||, ||P(n,n) - P(n-1,n-1)||')


%% Take K(1,0) = 10^4 and run the covariance and information kalman filters

% Preliminary Study for Covariance Kalman Filter: find num of iterations 
% needed to attain steady state covariance matrix

iter_count = 0;

% Initial x(0) (actual) as a vector with iid N(0,1) components
x_n_actual = randn(4,1);
% Initialize x_h[1|0] = 0
x_n_nm1 = zeros(4,1);
% Initialize K(1,0) to identity matrix
K_n_nm1 = 10^4;

while (norm(K_n_nm1-K1_ss)./norm(K1_ss)) > 0.01
        
    % Keep track of required num of KF iterations
    iter_count = iter_count + 1;

    % Generate v(n)
    unit_white = 1/sqrt(2)*(randn(4,1)); 
    v = (Qv^0.5)*unit_white; 
    % Generate x(n+1)
    x_n_actual = A1*x_n_actual + v;

    % Generate w(n)
    unit_white = 1/sqrt(2)*(randn(2,1)); 
    w = (Qw^0.5)*unit_white; 
    % Generate y(n)
    y = C*x_n_actual + w;

    % Standard Kalman Filter
    [x_n_n,x_np1_n,K_n_n,K_np1_n,G] = core_kalman_covar(x_n_nm1,K_n_nm1,y,A1,C,Qw,Qv);

    % Update values
    x_n_nm1 = x_np1_n; 
    K_n_nm1 = K_np1_n; 

end

num_iter_A1_KF = mean(iter_count) 

% Repeat for A2.
iter_count_A2 = 0;
 
% Initial x(0) (actual) is a vector with iid N(0,1) components
x_n_actual = randn(4,1);
% Initialize x_h[1|0] = 0
x_n_nm1 = zeros(4,1);
% Initialize K(1,0) to identity matrix
K_n_nm1 = 10^4;

while (norm(K_n_nm1 - K2_ss)./norm(K2_ss)) > 0.01
    % Keep track of required num of KF iterations
    iter_count_A2 = iter_count_A2 + 1;

    % Generate v(n)
    unit_white = 1/sqrt(2)*(randn(4,1)); 
    v = (Qv^0.5)*unit_white; 
    % Generate x(n+1)
    x_n_actual = A2*x_n_actual + v; 

    % Generate w(n)
    unit_white = 1/sqrt(2)*(randn(2,1)); 
    w = (Qw^0.5)*unit_white; 
    % Generate y(n)
    y = C*x_n_actual + w; 

    % Standard Kalman Filter
    [x_n_n,x_np1_n,K_n_n,K_np1_n,G] = core_kalman_covar(x_n_nm1,K_n_nm1,y,A2,C,Qw,Qv);

    % Update values
    x_n_nm1 = x_np1_n; 
    K_n_nm1 = K_np1_n; 
end

num_iter_A2_KF = mean(iter_count_A2)

% N1 and N2 increased by 1 each over beginning K(1,0) as the identity
% matrix. The error begins much higher but quickly corrects and is brought
% down.

%% Run KF and IKF routine

% Store general variables
x_actual = zeros(4,21);
y = zeros(2,20); 

% Store KF variables
x_n_nm1 = zeros(4,21);
x_n_n = zeros(4,21); 
norm_Khat_Kss = zeros(1,20);
norm_K_n_nm1 = zeros(1,20);
norm_Knn = zeros(1,20);
K_n_n = zeros(4,4*20);

% Store IKF variables
chi_n_nm1 = zeros(4,21); 
chi_n_n = zeros(4,21); 
norm_Phat_Pss = zeros(1,20);
norm_P_n_nm1 = zeros(1,20);
norm_Pnn = zeros(1,20);
P_n_n = zeros(4,4*20);

% Initial Conditions
% x(0) = vector with iid N(0,1) components
x_actual(:,1) = randn(4,1); 
% Initialize K(1,0) to 10^4, P(1,0) = inv(K(1,0))
K_n_nm1 = 10^4;
P_n_nm1 = inv(10^4);

% Generate v
unit_var_white = 1/sqrt(2)*(randn(4,20));
v = (Qv^0.5)*unit_var_white; 
% Generate w(n)
unit_white = 1/sqrt(2)*(randn(2,20)); 
w = (Qw^0.5)*unit_white; 

for n = 2:11
    % Process equation x(n+1) = A*x(n) + v(n) for N1 time steps using A1
    x_actual(:,n) = A1*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % STANDARD KF
    % Compute ||K(n,n-1) - K1_ss||
    norm_Khat_Kss(1,n-1) = norm(K_n_nm1 - K1_ss);
    % Compute ||K(n,n-1)||
    norm_K_n_nm1(1,n-1) = norm(K_n_nm1);
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [x_n_n(:,n-1),x_n_nm1(:,n),K_n_n(:,((n-1)*4)-3:(n-1)*4),K_np1_n,G] = core_kalman_covar(x_n_nm1(:,n-1),K_n_nm1,y(:,n-1),A1,C,Qw,Qv);
    % Compute ||K(n,n)||
    norm_Knn(1,n-1) = norm(K_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    K_n_nm1 = K_np1_n;
   
    % IKF
    % Compute ||P(n,n-1) - inv(K1_ss)||
    norm_Phat_Pss(1,n-1) = norm(P_n_nm1 - inv(K1_ss));
    % Compute ||P(n,n-1)||
    norm_P_n_nm1(1,n-1) = norm(P_n_nm1);
    % Run one iteration Kalman filter to approximate chi(n|n) and chi(n+1|n)
    [chi_n_n(:,n-1),chi_n_nm1(:,n),P_n_n(:,((n-1)*4)-3:(n-1)*4),P_np1_n] = core_kalman_covar(chi_n_nm1(:,n-1),P_n_nm1,y(:,n-1),A1,C,Qw,Qv);
    % Compute ||P(n,n)||
    norm_Pnn(1,n-1) = norm(P_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    P_n_nm1 = P_np1_n;
end

for n = 12:21
    % Process equation x(n+1) = A*x(n) + v(n) for N2 time steps using A2
    x_actual(:,n) = A2*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % STANDARD KF
    % Compute ||K(n,n-1) - K1_ss||
    norm_Khat_Kss(1,n-1) = norm(K_n_nm1 - K2_ss);
    % Compute ||K(n,n-1)||
    norm_K_n_nm1(1,n-1) = norm(K_n_nm1);
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [x_n_n(:,n-1),x_n_nm1(:,n),K_n_n(:,((n-1)*4)-3:(n-1)*4),K_np1_n,G] = core_kalman_covar(x_n_nm1(:,n-1),K_n_nm1,y(:,n-1),A2,C,Qw,Qv);
    % Compute ||K(n,n)||
    norm_Knn(1,n-1) = norm(K_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    K_n_nm1 = K_np1_n;
    
    % IKF
    % Compute ||P(n,n-1) - inv(K2_ss)||
    norm_Phat_Pss(1,n-1) = norm(P_n_nm1 - inv(K2_ss));
    % Compute ||P(n,n-1)||
    norm_P_n_nm1(1,n-1) = norm(P_n_nm1);
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [chi_n_n(:,n-1),chi_n_nm1(:,n),P_n_n(:,((n-1)*4)-3:(n-1)*4),P_np1_n] = core_kalman_covar(chi_n_nm1(:,n-1),P_n_nm1,y(:,n-1),A2,C,Qw,Qv);
    % Compute ||P(n,n)||
    norm_Pnn(1,n-1) = norm(P_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    P_n_nm1 = P_np1_n;
end

% Plot results

% Graph |x(n)|
figure
sgtitle('K(1,0) = 10^4')
subplot(3,4,1:2)
plot(0:10,vecnorm(x_actual(:,1:11)),'r.-')
hold on
plot(10:20,vecnorm(x_actual(:,11:21)),'b.-')
hold on
xline(10,'--')
title('|x(n)| = Euclidean Length of Actual State Variables')
xlabel('time[n]')
legend('A1','A2')

subplot(3,4,3:4)
plot3(x_actual(1,1:11),x_actual(2,1:11),x_actual(3,1:11),'r.-')
hold on
plot3(x_actual(1,11:21),x_actual(2,11:21),x_actual(3,11:21),'b.-')
legend('A1','A2')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('Plot of (x1,x2,x3) components of the actual state vector to visualize trajectory')

% Graph of | x(n) - x_h(n|n-1) | and | x(n) - x_h(n|n) | where n ranges
% from n=1 to n=20
norm_xpast = vecnorm(x_actual(:,2:21) - x_n_nm1(:,2:21));
norm_xpastandpres = vecnorm(x_actual(:,2:21) - x_n_n(:,1:20));
% Graph of | x(n) - chi(n|n-1) | and | x(n) - chi(n|n) | where n ranges
% from n=1 to n=20
norm_chi_past = vecnorm(x_actual(:,2:21) - chi_n_nm1(:,2:21));
norm_chi_pastandpres = vecnorm(x_actual(:,2:21) - chi_n_n(:,1:20));

subplot(3,4,5:8)
plot(1:20,norm_xpast,'m')
hold on
plot(1:20,norm_xpastandpres,'g')
hold on 
plot(1:20,norm_chi_past,'y')
hold on
plot(1:20,norm_chi_pastandpres,'b')
title('Error of Kalman Prediction and Kalman Filtering over time')
xlabel('time[n]')
ylabel('Euclidean Length of the Error')
legend('Standard Kalman Prediction','Standard Kalman Filtering','IKF Prediction','IKF Filtering')

subplot(3,4,9)
plot(1:10,norm_Khat_Kss(1:10),'r.--')
hold on
plot(11:20,norm_Khat_Kss(11:20),'b.--')
legend('||K-Kss|| when A1 used','||K-Kss|| when A2 used')
title('Error between K(n,n-1) and steady-state K')
xlabel('time[n]')

subplot(3,4,10)
plot(1:10,norm_Phat_Pss(1:10),'r.-')
hold on
plot(11:20,norm_Phat_Pss(11:20),'b.-')
legend('||P-Pss|| when A1 used','||P-Pss|| when A2 used')
title('Error between inv[K(n,n-1)]=P(n,n-1) and inv(Ki,ss) = Pi,ss')
xlabel('time[n]')

% Compute ||K(n,n) - K(n-1,n-1)||
% This sequence begins at K(2,2)-K(1,1) because K(0,0) doesnt exist
Knn_start2 = K_n_n(:,5:end); %remove K(1,1)
Knn_start1 = K_n_n(:,1:end-4); %remove K(20,20)
K_minus_Kprev = Knn_start2 - Knn_start1;
norm_K_Kprev = zeros(1,19);
for l=1:19
   norm_K_Kprev(1,l) = norm(K_minus_Kprev(:,(l*4)-3:l*4));
end

% Compute ||P(n,n) - P(n-1,n-1)||
% This sequence begins at P(2,2)-P(1,1) because P(0,0) doesnt exist
Pnn_start2 = P_n_n(:,5:end); %remove P(1,1)
Pnn_start1 = P_n_n(:,1:end-4); %remove P(20,20)
P_minus_Pprev = Pnn_start2 - Pnn_start1;
norm_P_Pprev = zeros(1,19);
for l=1:19
   norm_P_Pprev(1,l) = norm(P_minus_Pprev(:,(l*4)-3:l*4));
end

% Graph of ||K(n,n-1)||, ||K(n,n)||, ||K(n,n) - K(n-1,n-1)||
subplot(3,4,11)
plot(1:20,norm_K_n_nm1,'.-')
hold on
plot(1:20,norm_Knn,'.-')
hold on
plot(2:20,norm_K_Kprev,'.-')
legend('|| K(n,n-1) ||','|| K(n,n) ||','|| K(n,n) - K(n-1,n-1) ||')
title('K Norms')
xlabel('n')

subplot(3,4,12)
plot(1:20,norm_P_n_nm1,'.-')
hold on
plot(1:20,norm_Pnn,'.-')
hold on
plot(2:20,norm_P_Pprev,'.-')
legend('|| P(n,n-1) ||','|| P(n,n) ||','|| P(n,n) - P(n-1,n-1) ||')
xlabel('n')
title('P Norms')



%% Square Root Kalman Filter
% Preliminary Study for SQ Kalman Filter: find num of iterations 
% needed to attain steady state covariance matrix

iter_count_A1 = 0;

% Initial x(0) (actual) is a vector with iid N(0,1) components
x_n_actual = randn(4,1);
% Initialize K(1,0) to identity matrix
sqrt_K_nmin1 = eye(4);

while (norm(K_n_nm1^2 - K1_ss)./norm(K1_ss)) > 0.01
    % Keep track of required num of KF iterations
    iter_count_A1 = iter_count_A1 + 1;

    % Generate v(n)
    unit_white = 1/sqrt(2)*(randn(4,1)); 
    v = (Qv^0.5)*unit_white; 
    % Generate x(n+1)
    x_n_actual = A2*x_n_actual + v; 

    % Generate w(n)
    unit_white = 1/sqrt(2)*(randn(2,1)); 
    w = (Qw^0.5)*unit_white; 
    % Generate y(n)
    y = C*x_n_actual + w; 

    % Square Root Method
    [x_filt,x_pred,g,sqrt_r,sqrt_K_n] = core_kalman_sqrt(x_n_actual,y,A1,C,sqrt_K_nmin1,0.01)
    sqrt_K_nmin1 = sqrt_K_n; 
end

num_iter_A1_SQKF = mean(iter_count_A1)

%% Run KF and IKF routine

% Store general variables
x_actual = zeros(4,21);
y = zeros(2,20); 

% Store KF variables
x_n_nm1 = zeros(4,21);
x_n_n = zeros(4,21); 
norm_Khat_Kss = zeros(1,20);
norm_K_n_nm1 = zeros(1,20);
norm_Knn = zeros(1,20);
K_n_n = zeros(4,4*20);

% Store IKF variables
chi_n_nm1 = zeros(4,21); 
chi_n_n = zeros(4,21); 
norm_Phat_Pss = zeros(1,20);
norm_P_n_nm1 = zeros(1,20);
norm_Pnn = zeros(1,20);
P_n_n = zeros(4,4*20);

% Initial Conditions
% x(0) = vector with iid N(0,1) components
x_actual(:,1) = randn(4,1); 
% Initialize K(1,0) to 10^4, P(1,0) = inv(K(1,0))
K_n_nm1 = 10^4;
P_n_nm1 = inv(10^4);

% Generate v
unit_var_white = 1/sqrt(2)*(randn(4,20));
v = (Qv^0.5)*unit_var_white; 
% Generate w(n)
unit_white = 1/sqrt(2)*(randn(2,20)); 
w = (Qw^0.5)*unit_white; 

for n = 2:11
    % Process equation x(n+1) = A*x(n) + v(n) for N1 time steps using A1
    x_actual(:,n) = A1*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % STANDARD KF
    % Compute ||K(n,n-1) - K1_ss||
    norm_Khat_Kss(1,n-1) = norm(K_n_nm1 - K1_ss);
    % Compute ||K(n,n-1)||
    norm_K_n_nm1(1,n-1) = norm(K_n_nm1);
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [x_n_n(:,n-1),x_n_nm1(:,n),K_n_n(:,((n-1)*4)-3:(n-1)*4),K_np1_n,G] = core_kalman_covar(x_n_nm1(:,n-1),K_n_nm1,y(:,n-1),A1,C,Qw,Qv);
    % Compute ||K(n,n)||
    norm_Knn(1,n-1) = norm(K_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    K_n_nm1 = K_np1_n;
   
    % IKF
    % Compute ||P(n,n-1) - inv(K1_ss)||
    norm_Phat_Pss(1,n-1) = norm(P_n_nm1 - inv(K1_ss));
    % Compute ||P(n,n-1)||
    norm_P_n_nm1(1,n-1) = norm(P_n_nm1);
    % Run one iteration Kalman filter to approximate chi(n|n) and chi(n+1|n)
    [chi_n_n(:,n-1),chi_n_nm1(:,n),P_n_n(:,((n-1)*4)-3:(n-1)*4),P_np1_n] = core_kalman_covar(chi_n_nm1(:,n-1),P_n_nm1,y(:,n-1),A1,C,Qw,Qv);
    % Compute ||P(n,n)||
    norm_Pnn(1,n-1) = norm(P_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    P_n_nm1 = P_np1_n;
end

for n = 12:21
    % Process equation x(n+1) = A*x(n) + v(n) for N2 time steps using A2
    x_actual(:,n) = A2*x_actual(:,n-1) + v(:,n-1);
    % Observation equation y(n) = C*x(n) + w(n)
    y(:,n-1) = C*x_actual(:,n) + w(:,n-1);
    
    % STANDARD KF
    % Compute ||K(n,n-1) - K1_ss||
    norm_Khat_Kss(1,n-1) = norm(K_n_nm1 - K2_ss);
    % Compute ||K(n,n-1)||
    norm_K_n_nm1(1,n-1) = norm(K_n_nm1);
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [x_n_n(:,n-1),x_n_nm1(:,n),K_n_n(:,((n-1)*4)-3:(n-1)*4),K_np1_n,G] = core_kalman_covar(x_n_nm1(:,n-1),K_n_nm1,y(:,n-1),A2,C,Qw,Qv);
    % Compute ||K(n,n)||
    norm_Knn(1,n-1) = norm(K_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    K_n_nm1 = K_np1_n;
    
    % IKF
    % Compute ||P(n,n-1) - inv(K2_ss)||
    norm_Phat_Pss(1,n-1) = norm(P_n_nm1 - inv(K2_ss));
    % Compute ||P(n,n-1)||
    norm_P_n_nm1(1,n-1) = norm(P_n_nm1);
    % Run one iteration Kalman filter to approximate x(n|n) and x(n+1|n)
    [chi_n_n(:,n-1),chi_n_nm1(:,n),P_n_n(:,((n-1)*4)-3:(n-1)*4),P_np1_n] = core_kalman_covar(chi_n_nm1(:,n-1),P_n_nm1,y(:,n-1),A2,C,Qw,Qv);
    % Compute ||P(n,n)||
    norm_Pnn(1,n-1) = norm(P_n_n(:,((n-1)*4)-3:(n-1)*4));
    % Update values
    P_n_nm1 = P_np1_n;
end

% Plot results

% Graph |x(n)|
figure
sgtitle('K(1,0) = 10^4')
subplot(3,4,1:2)
plot(0:10,vecnorm(x_actual(:,1:11)),'r.-')
hold on
plot(10:20,vecnorm(x_actual(:,11:21)),'b.-')
hold on
xline(10,'--')
title('|x(n)| = Euclidean Length of Actual State Variables')
xlabel('time[n]')
legend('A1','A2')

subplot(3,4,3:4)
plot3(x_actual(1,1:11),x_actual(2,1:11),x_actual(3,1:11),'r.-')
hold on
plot3(x_actual(1,11:21),x_actual(2,11:21),x_actual(3,11:21),'b.-')
legend('A1','A2')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('Plot of (x1,x2,x3) components of the actual state vector to visualize trajectory')

% Graph of | x(n) - x_h(n|n-1) | and | x(n) - x_h(n|n) | where n ranges
% from n=1 to n=20
norm_xpast = vecnorm(x_actual(:,2:21) - x_n_nm1(:,2:21));
norm_xpastandpres = vecnorm(x_actual(:,2:21) - x_n_n(:,1:20));
% Graph of | x(n) - chi(n|n-1) | and | x(n) - chi(n|n) | where n ranges
% from n=1 to n=20
norm_chi_past = vecnorm(x_actual(:,2:21) - chi_n_nm1(:,2:21));
norm_chi_pastandpres = vecnorm(x_actual(:,2:21) - chi_n_n(:,1:20));

subplot(3,4,5:8)
plot(1:20,norm_xpast,'m')
hold on
plot(1:20,norm_xpastandpres,'g')
hold on 
plot(1:20,norm_chi_past,'y')
hold on
plot(1:20,norm_chi_pastandpres,'b')
title('Error of Kalman Prediction and Kalman Filtering over time')
xlabel('time[n]')
ylabel('Euclidean Length of the Error')
legend('Standard Kalman Prediction','Standard Kalman Filtering','IKF Prediction','IKF Filtering')

subplot(3,4,9)
plot(1:10,norm_Khat_Kss(1:10),'r.--')
hold on
plot(11:20,norm_Khat_Kss(11:20),'b.--')
legend('||K-Kss|| when A1 used','||K-Kss|| when A2 used')
title('Error between K(n,n-1) and steady-state K')
xlabel('time[n]')

subplot(3,4,10)
plot(1:10,norm_Phat_Pss(1:10),'r.-')
hold on
plot(11:20,norm_Phat_Pss(11:20),'b.-')
legend('||P-Pss|| when A1 used','||P-Pss|| when A2 used')
title('Error between inv[K(n,n-1)]=P(n,n-1) and inv(Ki,ss) = Pi,ss')
xlabel('time[n]')

% Compute ||K(n,n) - K(n-1,n-1)||
% This sequence begins at K(2,2)-K(1,1) because K(0,0) doesnt exist
Knn_start2 = K_n_n(:,5:end); %remove K(1,1)
Knn_start1 = K_n_n(:,1:end-4); %remove K(20,20)
K_minus_Kprev = Knn_start2 - Knn_start1;
norm_K_Kprev = zeros(1,19);
for l=1:19
   norm_K_Kprev(1,l) = norm(K_minus_Kprev(:,(l*4)-3:l*4));
end

% Compute ||P(n,n) - P(n-1,n-1)||
% This sequence begins at P(2,2)-P(1,1) because P(0,0) doesnt exist
Pnn_start2 = P_n_n(:,5:end); %remove P(1,1)
Pnn_start1 = P_n_n(:,1:end-4); %remove P(20,20)
P_minus_Pprev = Pnn_start2 - Pnn_start1;
norm_P_Pprev = zeros(1,19);
for l=1:19
   norm_P_Pprev(1,l) = norm(P_minus_Pprev(:,(l*4)-3:l*4));
end

% Graph of ||K(n,n-1)||, ||K(n,n)||, ||K(n,n) - K(n-1,n-1)||
subplot(3,4,11)
plot(1:20,norm_K_n_nm1,'.-')
hold on
plot(1:20,norm_Knn,'.-')
hold on
plot(2:20,norm_K_Kprev,'.-')
legend('|| K(n,n-1) ||','|| K(n,n) ||','|| K(n,n) - K(n-1,n-1) ||')
title('K Norms')
xlabel('n')

subplot(3,4,12)
plot(1:20,norm_P_n_nm1,'.-')
hold on
plot(1:20,norm_Pnn,'.-')
hold on
plot(2:20,norm_P_Pprev,'.-')
legend('|| P(n,n-1) ||','|| P(n,n) ||','|| P(n,n) - P(n-1,n-1) ||')
xlabel('n')
title('P Norms')



%% Extended Kalman Filter

% Setup Enviorenment

% 4 state variables and 4 measurements, i.e. x and y are both in R^4
% The process and observation equations are:
%       x(n+1) = f(x(n),n) + v(n)
%       y(n) = h(x(n),n) + w(n)
% where f and h apply a nonlinear operation, v and w are white noise with
% with Qv = 0.2I and Qw = 0.1*I
Qv_EKF = 0.2*eye(4);
Qw_EKF = 0.1*eye(4);

% Case 1: High degree of nonlinearity and h is non-monotonic
% Case 2: f is not differentiable

% Case 1:
% Anonymous Functions to apply nonlinear f and h to x. x is a 4x1 column vector
f_EKF = @(x) [10*pi*sin(x(1,1)*x(3,1)); 10*pi*sin(x(2,1)*x(4,1)); 10*pi*sin(x(3,1)); 10*pi*sin(x(4,1))];
h_EKF = @(x) sin(x);

% Use the process and observation equations to generate 100 time steps of x
% and y

x_EKF = zeros(4,101);
y_EKF = zeros(4,100);

% Initial Condition
x_EKF(:,1) = [1;1;1;1]; %x(0) = all ones

% Generate v
unit_var_white = 1/sqrt(2)*(randn(4,100));
v = (Qv_EKF^0.5)*unit_var_white; 
% Generate w
unit_white = 1/sqrt(2)*(randn(4,100)); 
w = (Qw_EKF^0.5)*unit_white; 

for n = 2:101
    % Process equation x(n+1) = f(x(n),n) + v(n)
    x_EKF(:,n) = f_EKF(x_EKF(:,n-1)) + v(:,n-1);
    % Observation equation y(n) = h(x(n),n) + w(n)
    y_EKF(:,n-1) = h_EKF(x_EKF(:,n)) + w(:,n-1);
   
end

% Plot State |x(n)| versus Time
figure
sgtitle('EKF')
subplot(2,2,1)
plot(0:100,vecnorm(x_EKF),'.-')
title('|x(n)| = Euclidean Length of Actual State Variables')
xlabel('time[n]')

subplot(2,2,2)
plot3(x_EKF(1,:),x_EKF(2,:),x_EKF(3,:),'r.-')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('Plot of (x1,x2,x3) components of the actual state vector to visualize trajectory')

subplot(2,2,3)
plot(1:100,vecnorm(y_EKF),'.-')
title('|y(n)| = Euclidean Length of Measurements y')
xlabel('time[n]')

subplot(2,2,4)
plot3(y_EKF(1,:),y_EKF(2,:),y_EKF(3,:),'r.-')
xlabel('y1')
ylabel('y2')
zlabel('y3')
title('Plot of (y1,y2,y3) measurement components to visualize trajectory of y')

% Clearly, the nonlinearity causes the state vector to bounce all over the
% place and not converge.

% Compute the Jacobians F(x) and H(x)
% The Jacobian matrix F(x) is NxN where N is the dimension of x (4-by-4 in
% our case). To compute the Jacobian, take the partial derivative of the f
% equation with respect to each of the components of x/state variables:
% F(x) = del f/del(x1,x2,x3,x4).
% del f/del x1 = [10*pi*x(3,1)*cos(x(1,1)*x(3,1)); 0; 0; 0]
% del f/del x2 = [0; 10*pi*x(4,1)*cos(x(2,1)*x(4,1)); 0; 0]
% del f/del x3 = [10*pi*x(1,1)*cos(x(1,1)*x(3,1)); 0; 10*pi*cos(x(3,1)); 0]
% del f/del x4 = [0; 10*pi*x(2,1)*cos(x(2,1)*x(4,1)); 0; 10*pi*cos(x(4,1))]
F_EKF = @(x) [10*pi*x(3,1)*cos(x(1,1)*x(3,1)) 0 10*pi*x(1,1)*cos(x(1,1)*x(3,1)) 0; 0 10*pi*x(4,1)*cos(x(2,1)*x(4,1)) 0 10*pi*x(2,1)*cos(x(2,1)*x(4,1)); 0 0 10*pi*cos(x(3,1)) 0; 0 0 0 10*pi*cos(x(4,1))];

% H(x) = del h/del(x1,x2,x3,x4). h = [sin(x1) sin(x2) sin(x3) sin(x4)]
% del h/del x1 = [cos(x1); 0; 0; 0]
% del h/del x2 = [0; cos(x2); 0; 0]
% del h/del x3 = [0; 0; cos(x3); 0]
% del h/del x4 = [0; 0; 0; cos(x4)]
H_EKF = @(x) [cos(x(1,1)) 0 0 0; 0 cos(x(2,1)) 0 0; 0 0 cos(x(3,1)) 0; 0 0 0 cos(x(4,1))];

%% Run EKF 100 times
x_nn_EKF = zeros(4,101);
x_nn_EKF(:,1) = [0;0;0;0];
K_nm1_nm1 = eye(4);
for n = 1:100
    [x_nn_EKF(:,n+1),K_nn] = core_EFK1(x_nn_EKF(:,n),K_nm1_nm1,y_EKF(:,n),Qw_EKF,Qv_EKF);
    %update
    K_nm1_nm1 = K_nn;
end 

% Plot ||x(n)-x_h(n|n)||
dif = vecnorm(x_EKF - x_nn_EKF);
figure
plot(1:100,dif(2:end))
title('||x(n)-x_h(n|n)|| for EKF (h(x) is NOT monotonic)')
xtitle('time[n]')

% Performs badly, oscillates.

%% Change h(x) to be tan^-1(x) and repeat EKF experiment
h_EKF = @(x) atan(x); 

% Use the process and observation equations to generate 100 time steps of x
% and y
x_EKF_h2 = zeros(4,101);
y_EKF_h2 = zeros(4,100);

% Initial Condition
x_EKF_h2(:,1) = [1;1;1;1]; %x(0) = all ones

% Generate v
unit_var_white = 1/sqrt(2)*(randn(4,100));
v = (Qv_EKF^0.5)*unit_var_white; 
% Generate w
unit_white = 1/sqrt(2)*(randn(4,100)); 
w = (Qw_EKF^0.5)*unit_white; 

for n = 2:101
    % Process equation x(n+1) = f(x(n),n) + v(n)
    x_EKF_h2(:,n) = f_EKF(x_EKF_h2(:,n-1)) + v(:,n-1);
    % Observation equation y(n) = h(x(n),n) + w(n)
    y_EKF_h2(:,n-1) = h_EKF(x_EKF_h2(:,n)) + w(:,n-1);
   
end

% Plot State |x(n)| versus Time
figure
sgtitle('EKF h=monotonic')
subplot(2,2,1)
plot(0:100,vecnorm(x_EKF_h2),'.-')
title('|x(n)| = Euclidean Length of Actual State Variables')
xlabel('time[n]')

subplot(2,2,2)
plot3(x_EKF_h2(1,:),x_EKF_h2(2,:),x_EKF_h2(3,:),'r.-')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('Plot of (x1,x2,x3) components of the actual state vector to visualize trajectory')

subplot(2,2,3)
plot(1:100,vecnorm(y_EKF_h2),'.-')
title('|y(n)| = Euclidean Length of Measurements y')
xlabel('time[n]')

subplot(2,2,4)
plot3(y_EKF_h2(1,:),y_EKF_h2(2,:),y_EKF_h2(3,:),'r.-')
xlabel('y1')
ylabel('y2')
zlabel('y3')
title('Plot of (y1,y2,y3) measurement components to visualize trajectory of y')


% Run EKF 100 times
x_nn_EKF_h2 = zeros(4,101);
x_nn_EKF_h2(:,1) = [0;0;0;0];
K_nm1_nm1_h2 = eye(4);
for n = 1:100
    [x_nn_EKF_h2(:,n+1),K_nn_h2] = core_EFK1_h2(x_nn_EKF_h2(:,n),K_nm1_nm1_h2,y_EKF_h2(:,n),Qw_EKF,Qv_EKF);
    %update
    K_nm1_nm1_h2 = K_nn_h2;
end 

% Plot ||x(n)-x_h(n|n)||
dif2 = vecnorm(x_EKF_h2 - x_nn_EKF_h2);
figure
plot(1:100,dif2(2:end))
title('||x(n)-x_h(n|n)|| for EKF when h(x) = tan^-1(x) [monotonic]')
xtitle('time[n]')

% Not too much improvement.

%% Case 2: f is not differentiable

% Process Equation:
% x(n+1) = g(A1*x(n)) + v(n)

% Anonymous Functions
f_EKF_C2 = @(x) abs(A1*x);
h_EKF_C2 = @(x) atan(x);

% Use the process and observation equations to generate 100 time steps of x
% and y
x_EKF_C2 = zeros(4,101);
y_EKF_C2 = zeros(4,100);
x_EKF_C2(:,1) = [1;1;1;1]; %x(0) = all ones

% Generate v
unit_var_white = 1/sqrt(2)*(randn(4,100));
v = (Qv_EKF^0.5)*unit_var_white; 
% Generate w
unit_white = 1/sqrt(2)*(randn(4,100)); 
w = (Qw_EKF^0.5)*unit_white; 

for n = 2:101
    % Process equation x(n+1) = f(x(n)) + v(n) = g(A1*x(n)) + v(n)
    x_EKF_C2(:,n) = f_EKF_C2(x_EKF_C2(:,n-1)) + v(:,n-1);
    % Observation equation y(n) = h(x(n),n) + w(n)
    y_EKF_C2(:,n-1) = h_EKF_C2(x_EKF_C2(:,n)) + w(:,n-1); 
end

% Plot State |x(n)| versus Time
figure
sgtitle('EKF Case2: Nondifferentiable f')
subplot(2,2,1)
plot(0:100,vecnorm(x_EKF_C2),'.-')
title('|x(n)| = Euclidean Length of Actual State Variables')
xlabel('time[n]')

subplot(2,2,2)
plot3(x_EKF_C2(1,:),x_EKF_C2(2,:),x_EKF_C2(3,:),'r.-')
xlabel('x1')
ylabel('x2')
zlabel('x3')
title('Plot of (x1,x2,x3) components of the actual state vector to visualize trajectory')


subplot(2,2,3)
plot(1:100,vecnorm(y_EKF_C2),'.-')
title('|y(n)| = Euclidean Length of Measurements y')
xlabel('time[n]')

subplot(2,2,4)
plot3(y_EKF_C2(1,:),y_EKF_C2(2,:),y_EKF_C2(3,:),'r.-')
xlabel('y1')
ylabel('y2')
zlabel('y3')
title('Plot of (y1,y2,y3) measurement components to visualize trajectory of y')

% Compute the Jacobian F(x)
% F(x) = del f/del(x1,x2,x3,x4). f = g(A1(x))
% A1(x) elementwise opens up to be a vector = 
% [-0.9x1 + x2       ]
% [-0.9x2            ]
% [-0.5x3 + 0.5x4    ]
% [x1 - 0.5x3 - 0.5x4]
% Therefore, f = 
% [g(-0.9x1 + x2); g(-0.9x2); g(-0.5x3 + 0.5x4); g(x1 - 0.5x3 - 0.5x4)]

% To compute the Jacobian,
% del f/del x1 = [-0.9 * g'(-0.9x1 + x2); 0; 0; g'(x1 - 0.5x3 - 0.5x4)]
% del f/del x2 = [g'(-0.9x1 + x2); -0.9 * g'(-0.9x2); 0; 0]
% del f/del x3 = [0; 0; -0.5 * g'(-0.5x3 + 0.5x4); -0.5 * g'(x1 - 0.5x3 - 0.5x4)]
% del f/del x4 = [0; 0; 0.5 * g'(-0.5x3 + 0.5x4); -0.5 * g'(x1 - 0.5x3 - 0.5x4)]
% In every place where there is a g', evaluate what is inside and apply g^d
% function = 1 if arg>0.1, -1 if arg<-0.1, 0 if between -0.1 and 0.1
% inclusive.
F_EKF_C2 = @(x) [-0.9*gprime(-0.9*x(1,1) + x(2,1)) gprime(-0.9*x(1,1) + x(2,1)) 0 0; 0 -0.9*gprime(-0.9*x(2,1)) 0 0; 0 0 -0.5*gprime(-0.5*x(3,1) + 0.5*x(4,1)) 0.5*gprime(-0.5*x(3,1) + 0.5*x(4,1)); gprime(x(1,1) - 0.5*x(3,1) - 0.5*x(4,1)) 0  -0.5*gprime(x(1,1) - 0.5*x(3,1) - 0.5*x(4,1)) -0.5*gprime(x(1,1) - 0.5*x(3,1) - 0.5*x(4,1))];
H_EKF_C2 = @(x) [1./(1+(x(1,1).^2)) 0 0 0; 0 1./(1+(x(2,1).^2)) 0 0; 0 0 1./(1+(x(3,1).^2)) 0; 0 0 0 1./(1+(x(4,1).^2))];

%% Run EKF Case2 100 times
x_nn_EKF_C2 = zeros(4,101);
x_nn_EKF_C2(:,1) = [0;0;0;0];
K_nm1_nm1_C2 = eye(4);
for n = 1:100
    [x_nn_EKF_C2(:,n+1),K_nn_C2] = core_EFK2(x_nn_EKF_C2(:,n),K_nm1_nm1_C2,y_EKF_C2(:,n),Qw_EKF,Qv_EKF,A1);
    %update
    K_nm1_nm1_C2 = K_nn_C2;
end 

% Plot ||x(n)-x_h(n|n)||
dif_C2 = vecnorm(x_EKF_C2 - x_nn_EKF_C2);
figure
plot(1:100,dif_C2(2:end))
title('||x(n)-x_h(n|n)|| for EKF Case 2: f is not differentiable')
xlabel('time[n]')

% Performs well.

%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Covariance Kalman Filter - core function to run one iteration of the filter

% Core function that runs one iteration of the Kalman filter. 
% Inputs are x_h(n|n-1) [refer to as x_in], 
%            K(n,n-1) [refer to as K_in]
% Outputs are x_h(n|n) [refer to as x_filt], 
%             x_h(n+1|n) [refer to as x_pred], 
%             K(n,n), 
%             K(n+1,n) [refer to as K_futn_n], 
%             G(n)

function [x_filt,x_pred,K_n_n,K_futn_n,G] = core_kalman_covar(x_in,K_in,y,A,C,Qw,Qv)
    % Compute Kalman gain = K(n,n-1)*C^H*(C*K(n,n-1)*C^H + Qw)^-1
    G = K_in*C'*inv(C*K_in*C' + Qw);
    
    % Compute Kalman innovations = y(n) - C*x(n|n-1) =  white noise by orthogonality
    alpha = y - C*x_in;
    
    % Compute filtering x = x_h(n|n) = x_h(n|n-1) + G*alpha
    x_filt = x_in + G*alpha;
    % Compute prediction x = x_h(n+1|n) = A*x_h(n|n)
    x_pred = A*x_filt;
    
    % Update K values for next iteration
    K_n_n = K_in - G*C*K_in;
    K_futn_n = A*K_n_n*A' + Qv;
end


%% Information Kalman Filter - core function to run one iteration of the filter

% Core function that runs one iteration of the information kalman filter. 
% Inputs are chi(n,n-1) [refer to as chi_in], 
%            P(n,n-1) [refer to as P_in]
% Outputs are chi(n,n) [refer to as chi_filt], 
%             chi(n+1,n) [refer to as chi_pred], 
%             P(n,n) [refer to as Pnn], 
%             P(n+1,n) [refer to as P_futn_n], 


function [chi_filt,chi_pred,Pnn,P_futn_n] = core_kalman_info(chi_in,P_in,y,A,C,Qw,Qv)
    % P(n,n) = P(n,n-1) + C^H*Qw^-1*C
    Pnn = P_in + C'*inv(Qw)*C;

    % M(n) = A^(-H)*P(n,n)*A^(-1)
    M = inv(A)'*Pnn*inv(A);

    % F(n) = (I + M*Qv)^-1
    F = inv(eye(4) + M*Qv);

    % P(n+1,n) = F*M
    P_futn_n = F*M;
    
    % chi_filt = chi(n,n) = chi(n,n-1) + C^H*Qw^-1*y(n)
    chi_filt = chi_in + C'*inv(Qw)*y;
    % chi_pred = chi(n+1,n) = F*A^-H*chi_filt
    chi_pred = F*inv(A)'*chi_filt;
end

%% Square Root Kalman Filter - core function to run one iteration of the filter

function [x_filt,x_pred,g,sqrt_r,sqrt_K_n] = core_kalman_sqrt(u,y,A,C,sqrt_K_nmin1,lamda)
    l = lamda^-0.5;
    [M,~] = size(u);
    
    % Setup prearray
    prearray = [1 u'*sqrt_K_nmin1; 0 l*sqrt_K_nmin1];
    
    % Annihilation using qr decomposition
    [Q,R] = qr(prearray'); %hermitian transpose A first because qr outputs an upper
    % triangular matrix R and we are looking for a lower triangular matrix.
    % By applying the hermitian transpose to R, we can obtain what we
    % require.
    
    % Postarray Extraction
    postarray = R';
 
    sqrt_r = postarray(1,1); %upper left corner
    sqrt_K_n = postarray(2,2:end);
    g = postarray(2:end,1)./sqrt_r;
        
    % Compute Kalman innovations = y(n) - C*x(n|n-1) =  white noise by orthogonality
    alpha = y - C*x_in;
    
    % Compute filtering x = x_h(n|n) = x_h(n|n-1) + G*alpha
    x_filt = x_in + g*alpha;
    % Compute prediction x = x_h(n+1|n) = A*x_h(n|n)
    x_pred = A*x_filt;
end 

%% Extended Kalman Filter - core function to run one iteration of the filter

function [x_nn,K_nn] = core_EFK1(x_nm1_nm1,K_nm1_nm1,y,Qw,Qv)
    % Anonymous functions for case1
    f_EKF = @(x) [10*pi*sin(x(1,1)*x(3,1)); 10*pi*sin(x(2,1)*x(4,1)); 10*pi*sin(x(3,1)); 10*pi*sin(x(4,1))];
    h_EKF = @(x) sin(x);
    F_EKF = @(x) [10*pi*x(3,1)*cos(x(1,1)*x(3,1)) 0 10*pi*x(1,1)*cos(x(1,1)*x(3,1)) 0; 0 10*pi*x(4,1)*cos(x(2,1)*x(4,1)) 0 10*pi*x(2,1)*cos(x(2,1)*x(4,1)); 0 0 10*pi*cos(x(3,1)) 0; 0 0 0 10*pi*cos(x(4,1))];
    H_EKF = @(x) [cos(x(1,1)) 0 0 0; 0 cos(x(2,1)) 0 0; 0 0 cos(x(3,1)) 0; 0 0 0 cos(x(4,1))];

    x_n_projprev = f_EKF(x_nm1_nm1);
    
    F_nminus1 = F_EKF(x_nm1_nm1);
    H_jac = H_EKF(x_n_projprev); %may have to be little h
    
    K_n_projprev = F_nminus1*K_nm1_nm1*F_nminus1' + Qv;
    alpha = y - h_EKF(x_n_projprev);
    S = H_jac*K_n_projprev*H_jac' + Qw;
    G = K_n_projprev*H_jac'*inv(S);
    
    x_nn = x_n_projprev + G*alpha;
    K_nn = (eye(4)-G*H_jac)*K_n_projprev;   
end

% Change h(x) to be the monotonic arctan(x)
function [x_nn,K_nn] = core_EFK1_h2(x_nm1_nm1,K_nm1_nm1,y,Qw,Qv)
    % Anonymous functions for case1
    f_EKF = @(x) [10*pi*sin(x(1,1)*x(3,1)); 10*pi*sin(x(2,1)*x(4,1)); 10*pi*sin(x(3,1)); 10*pi*sin(x(4,1))];
    h_EKF = @(x) atan(x); % Now h is monotonic 
    F_EKF = @(x) [10*pi*x(3,1)*cos(x(1,1)*x(3,1)) 0 10*pi*x(1,1)*cos(x(1,1)*x(3,1)) 0; 0 10*pi*x(4,1)*cos(x(2,1)*x(4,1)) 0 10*pi*x(2,1)*cos(x(2,1)*x(4,1)); 0 0 10*pi*cos(x(3,1)) 0; 0 0 0 10*pi*cos(x(4,1))];
    % derivative of arctan(x) = 1./(1+x^2)
    H_EKF = @(x) [1./(1+(x(1,1).^2)) 0 0 0; 0 1./(1+(x(2,1).^2)) 0 0; 0 0 1./(1+(x(3,1).^2)) 0; 0 0 0 1./(1+(x(4,1).^2))];

    x_n_projprev = f_EKF(x_nm1_nm1);
    
    F_nminus1 = F_EKF(x_nm1_nm1);
    H_jac = H_EKF(x_n_projprev); %may have to be little h
    
    K_n_projprev = F_nminus1*K_nm1_nm1*F_nminus1' + Qv;
    alpha = y - h_EKF(x_n_projprev);
    S = H_jac*K_n_projprev*H_jac' + Qw;
    G = K_n_projprev*H_jac'*inv(S);
    
    x_nn = x_n_projprev + G*alpha;
    K_nn = (eye(4)-G*H_jac)*K_n_projprev;   
end

% Case 2 f not differentiable
function [x_nn,K_nn] = core_EFK2(x_nm1_nm1,K_nm1_nm1,y,Qw,Qv,A1)
    % Anonymous functions for case2
    f_EKF = @(x) abs(A1*x);
    h_EKF = @(x) atan(x);
    F_EKF = @(x) [-0.9*gprime(-0.9*x(1,1) + x(2,1)) gprime(-0.9*x(1,1) + x(2,1)) 0 0; 0 -0.9*gprime(-0.9*x(2,1)) 0 0; 0 0 -0.5*gprime(-0.5*x(3,1) + 0.5*x(4,1)) 0.5*gprime(-0.5*x(3,1) + 0.5*x(4,1)); gprime(x(1,1) - 0.5*x(3,1) - 0.5*x(4,1)) 0  -0.5*gprime(x(1,1) - 0.5*x(3,1) - 0.5*x(4,1)) -0.5*gprime(x(1,1) - 0.5*x(3,1) - 0.5*x(4,1))];
    H_EKF = @(x) [1./(1+(x(1,1).^2)) 0 0 0; 0 1./(1+(x(2,1).^2)) 0 0; 0 0 1./(1+(x(3,1).^2)) 0; 0 0 0 1./(1+(x(4,1).^2))];

    x_n_projprev = f_EKF(x_nm1_nm1);
    
    F_nminus1 = F_EKF(x_nm1_nm1);
    H_jac = H_EKF(x_n_projprev); %may have to be little h
    
    K_n_projprev = F_nminus1*K_nm1_nm1*F_nminus1' + Qv;
    alpha = y - h_EKF(x_n_projprev);
    S = H_jac*K_n_projprev*H_jac' + Qw;
    G = K_n_projprev*H_jac'*inv(S);
    
    x_nn = x_n_projprev + G*alpha;
    K_nn = (eye(4)-G*H_jac)*K_n_projprev;   
end

%% Function to Apply g^d(arg) for EKF case 2 jacobian of non-differentiable f
function gd = gprime(arg)
    if arg > 0.1
        gd = 1
    elseif arg < -0.1
        gd = -1
    else 
        gd = 0
    end
end