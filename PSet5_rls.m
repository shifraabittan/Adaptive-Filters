%% Shifra Abittan
% Prof Fontaine
% ECE416 Adaptive Filters
% PSet5 RLS, QRD-RLS, and Order Recursive Algorithms

%% 1. Orthogonality Principle

% Prove that e(n,n-1) is orthogonal to v(n).
% e(n,n-1) is defined as x(n) - xh(n|n-1). The first term x(n) is orthogonal
% to v(n) because the present x(n) includes only PAST v values, i.e. 
% x(n) = Ax(n-1) + v(n-1). The v values are uncorrelated with each other.
% The second term of e(n,n-1) is xh(n|n-1). This is the projection of x(n)
% onto the span of y(n) from n=1 to n=n-1. y(n-1) = C*x(n-1) + w(n-1). Once
% again, the x(n-1) values includes only v up to n-2, so the first term
% C*x(n-1) is orthogonal to v and w and v are uncorrelated. 

% In summary, because each x depends only on PAST v values and not the
% present v value, e(n,n-1) is orthogonal to v.

% Prove that e(n,n-1) is orthogonal to w(n). 
% e(n,n-1) is defined as x(n) - xh(n|n-1). x(n) is composed of past x
% values and v values. w by definition is uncorrelated with v and with x(0)
% so it is uncorrelated with all future x values. xh(n|n-1) is composed of
% x values up to x(n-1) and w values up to w(n-1). w by definition is
% uncorrelated with all other w values. Therefore, e(n,n-1) is orthogonal
% to w(n).

%% 2. RLS

% Implement RLS to build an adaptive equalizer for the environment described
% in Problem Set 3. Two different pole pairs were presented, which results
% in two different difference equation. Both cases will be considered here.
% The particular case can be selected in the function that generates the
% u[n] and d[n] signals.

lamda = 0.95;
delta = 0.005;

% M=6

J_ktzi_C1 = zeros(100,num_iters);
J_e_C1 = zeros(100,num_iters);

% Case 1
for MonteCarlo = 1:100 %run the experiment 100 times so that can average Js
    
    num_iters = 100; %tune until algo converges and P(0) negligable

    [u,d] = u_and_d(num_iters+5,1); %generate u[n] and d[n] data
    w = zeros(6,num_iters+1); %initialize w(0) = 0, index off by one bc w(0) in first location
    P = (delta^-1) .* eye(6); %initialize P(0) = inv(delta)*I, i.e. phi(0) = del*I

    for it = 1:num_iters

        % RLS Equations
        % u_6(n) = [u(n) u(n-1) ... u(n-5)]'
        u_n = flipud(u(it:it+5,1));
        d_n = d(it+5,1); %line up d(n) and u(n)
        % pi(n) = P(n-1)*u(n)
        pi_n = P*u_n;
        % Gain Vector: k(n) = (lamda + u^H(n)*pi(n))^-1 * pi(n)
        k_n = inv(lamda + u_n'*pi_n)*pi_n;
        % A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        ktzi_n = d(it+5,1) - w(:,it)'*u_n; 
        % w(n) = w(n-1) + k(n)*conj(ktzi(n))
        w(:,it+1) = w(:,it) + k_n*conj(ktzi_n);
        % Prepare for next iteration: P(n) = inv(lamda)*P(n-1) - inv(lamda)*k(n)*pi^H(n)
        P = inv(lamda)*P - inv(lamda)*k_n*pi_n';

        % Learning Curves
        % Before Update/A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        % J'(n) = E(|ktzi(n)|^2)
        J_ktzi_C1(MonteCarlo,it) = abs(ktzi_n).^2;
        % After Update/A-Posteriori Estimation Error: e(n) = d(n) - w^H(n)*u(n)
        % J(n) = E(|e(n)|^2)
        e_n = d(it+5,1) - w(:,it+1)'*u_n;
        J_e_C1(MonteCarlo,it) = abs(e_n).^2;
    end 
end

% Case 2
J_ktzi_C2 = zeros(100,num_iters);
J_e_C2 = zeros(100,num_iters);

for MonteCarlo = 1:100 %run the experiment 100 times so that can average Js
    
    num_iters = 100; %tune until algo converges and P(0) negligable

    [u,d] = u_and_d(num_iters+5,2); %generate u[n] and d[n] data
    w = zeros(6,num_iters+1); %initialize w(0) = 0, index off by one bc w(0) in first location
    P = (delta^-1) .* eye(6); %initialize P(0) = inv(delta)*I, i.e. phi(0) = del*I

    for it = 1:num_iters

        % RLS Equations
        % u_6(n) = [u(n) u(n-1) ... u(n-5)]'
        u_n = flipud(u(it:it+5,1));
        % pi(n) = P(n-1)*u(n)
        pi_n = P*u_n;
        % Gain Vector: k(n) = (lamda + u^H(n)*pi(n))^-1 * pi(n)
        k_n = inv(lamda + u_n'*pi_n)*pi_n;
        % A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        ktzi_n = d(it+5,1) - w(:,it)'*u_n;
        % w(n) = w(n-1) + k(n)*conj(ktzi(n))
        w(:,it+1) = w(:,it) + k_n*conj(ktzi_n);
        % Prepare for next iteration: P(n) = inv(lamda)*P(n-1) - inv(lamda)*k(n)*pi^H(n)
        P = inv(lamda)*P - inv(lamda)*k_n*pi_n';

        % Learning Curves
        % Before Update/A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        % J'(n) = E(|ktzi(n)|^2)
        J_ktzi_C2(MonteCarlo,it) = abs(ktzi_n).^2;
        % After Update/A-Posteriori Estimation Error: e(n) = d(n) - w^H(n)*u(n)
        % J(n) = E(|e(n)|^2)
        e_n = d(it+5,1) - w(:,it+1)'*u_n;
        J_e_C2(MonteCarlo,it) = abs(e_n).^2;
    end 
end

% Graph RLS Results
figure
sgtitle('RLS M=6: Case 1 = poles 0.3 and 0.5, Case 2 = poles 0.3 and 0.95')
subplot(2,2,2)
plot(1:num_iters,J_e_C1(1,:))
hold on
plot(1:num_iters,J_ktzi_C1(1,:))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('J Learning Curves for One Simulation of RLS Case 1')
xlabel('adaptive iteration number')

subplot(2,2,1)
plot(1:num_iters,mean(J_e_C1))
hold on
plot(1:num_iters,mean(J_ktzi_C1))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('Average J Learning Curves for 100 Simulations of RLS Case 1')
xlabel('adaptive iteration number')

subplot(2,2,4)
plot(1:num_iters,J_e_C2(1,:))
hold on
plot(1:num_iters,J_ktzi_C2(1,:))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('J Learning Curves for One Simulation of RLS Case 2')
xlabel('adaptive iteration number')

subplot(2,2,3)
plot(1:num_iters,mean(J_e_C2))
hold on
plot(1:num_iters,mean(J_ktzi_C2))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('Average J Learning Curves for 100 Simulations of RLS Case 2')
xlabel('adaptive iteration number')

% As expected, the a-priori learning curve values for small n are extremely
% large but drop off quickly. The a-posteriori learning curve values for
% small n are very small but grow a little more. Overall, both curves
% converge extremely quickly. When n<=M, the a-postreiori error is exactly
% zero as expected from RLS.

%% M=3, too few tap weights

J_ktzi_C1 = zeros(100,num_iters);
J_e_C1 = zeros(100,num_iters);

% Case 1
for MonteCarlo = 1:100 %run the experiment 100 times so that can average Js
    
    num_iters = 100; %tune until algo converges and P(0) negligable

    [u,d] = u_and_d(num_iters+2,1); %generate u[n] and d[n] data
    w = zeros(3,num_iters+1); %initialize w(0) = 0, index off by one bc w(0) in first location
    P = (delta^-1) .* eye(3); %initialize P(0) = inv(delta)*I, i.e. phi(0) = del*I

    for it = 1:num_iters

        % RLS Equations
        % u_3(n) = [u(n) u(n-1) u(n-2)]'
        u_n = flipud(u(it:it+2,1));
        % pi(n) = P(n-1)*u(n)
        pi_n = P*u_n;
        % Gain Vector: k(n) = (lamda + u^H(n)*pi(n))^-1 * pi(n)
        k_n = inv(lamda + u_n'*pi_n)*pi_n;
        % A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        ktzi_n = d(it+2,1) - w(:,it)'*u_n;
        % w(n) = w(n-1) + k(n)*conj(ktzi(n))
        w(:,it+1) = w(:,it) + k_n*conj(ktzi_n);
        % Prepare for next iteration: P(n) = inv(lamda)*P(n-1) - inv(lamda)*k(n)*pi^H(n)
        P = inv(lamda)*P - inv(lamda)*k_n*pi_n';

        % Learning Curves
        % Before Update/A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        % J'(n) = E(|ktzi(n)|^2)
        J_ktzi_C1(MonteCarlo,it) = abs(ktzi_n).^2;
        % After Update/A-Posteriori Estimation Error: e(n) = d(n) - w^H(n)*u(n)
        % J(n) = E(|e(n)|^2)
        e_n = d(it+2,1) - w(:,it+1)'*u_n;
        J_e_C1(MonteCarlo,it) = abs(e_n).^2;
    end 
end

% Case 2
J_ktzi_C2 = zeros(100,num_iters);
J_e_C2 = zeros(100,num_iters);

for MonteCarlo = 1:100 %run the experiment 100 times so that can average Js
    
    num_iters = 100; %tune until algo converges and P(0) negligable

    [u,d] = u_and_d(num_iters+2,2); %generate u[n] and d[n] data
    w = zeros(3,num_iters+1); %initialize w(0) = 0, index off by one bc w(0) in first location
    P = (delta^-1) .* eye(3); %initialize P(0) = inv(delta)*I, i.e. phi(0) = del*I

    for it = 1:num_iters

        % RLS Equations
        % u_3(n) = [u(n) u(n-1) u(n-2)]'
        u_n = flipud(u(it:it+2,1));
        % pi(n) = P(n-1)*u(n)
        pi_n = P*u_n;
        % Gain Vector: k(n) = (lamda + u^H(n)*pi(n))^-1 * pi(n)
        k_n = inv(lamda + u_n'*pi_n)*pi_n;
        % A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        ktzi_n = d(it+2,1) - w(:,it)'*u_n;
        % w(n) = w(n-1) + k(n)*conj(ktzi(n))
        w(:,it+1) = w(:,it) + k_n*conj(ktzi_n);
        % Prepare for next iteration: P(n) = inv(lamda)*P(n-1) - inv(lamda)*k(n)*pi^H(n)
        P = inv(lamda)*P - inv(lamda)*k_n*pi_n';

        % Learning Curves
        % Before Update/A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        % J'(n) = E(|ktzi(n)|^2)
        J_ktzi_C2(MonteCarlo,it) = abs(ktzi_n).^2;
        % After Update/A-Posteriori Estimation Error: e(n) = d(n) - w^H(n)*u(n)
        % J(n) = E(|e(n)|^2)
        e_n = d(it+2,1) - w(:,it+1)'*u_n;
        J_e_C2(MonteCarlo,it) = abs(e_n).^2;
    end 
end

% Graph RLS Results
figure
sgtitle('RLS M=3 Too Few Tap Weights: Case 1 = poles 0.3 and 0.5, Case 2 = poles 0.3 and 0.95')
subplot(2,2,2)
plot(1:num_iters,J_e_C1(1,:))
hold on
plot(1:num_iters,J_ktzi_C1(1,:))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('J Learning Curves for One Simulation of RLS Case 1')
xlabel('adaptive iteration number')

subplot(2,2,1)
plot(1:num_iters,mean(J_e_C1))
hold on
plot(1:num_iters,mean(J_ktzi_C1))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('Average J Learning Curves for 100 Simulations of RLS Case 1')
xlabel('adaptive iteration number')

subplot(2,2,4)
plot(1:num_iters,J_e_C2(1,:))
hold on
plot(1:num_iters,J_ktzi_C2(1,:))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('J Learning Curves for One Simulation of RLS Case 2')
xlabel('adaptive iteration number')

subplot(2,2,3)
plot(1:num_iters,mean(J_e_C2))
hold on
plot(1:num_iters,mean(J_ktzi_C2))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('Average J Learning Curves for 100 Simulations of RLS Case 2')
xlabel('adaptive iteration number')

% Performs very well, despite having too few tap weights because RLS is
% much better than LMS.

%% M=10, too many tap weights

J_ktzi_C1 = zeros(100,num_iters);
J_e_C1 = zeros(100,num_iters);

% Case 1
for MonteCarlo = 1:100 %run the experiment 100 times so that can average Js
    
    num_iters = 100; %tune until algo converges and P(0) negligable

    [u,d] = u_and_d(num_iters+9,1); %generate u[n] and d[n] data
    w = zeros(10,num_iters+1); %initialize w(0) = 0, index off by one bc w(0) in first location
    P = (delta^-1) .* eye(10); %initialize P(0) = inv(delta)*I, i.e. phi(0) = del*I

    for it = 1:num_iters

        % RLS Equations
        % u_10(n) = [u(n) u(n-1) ... u(n-9)]'
        u_n = flipud(u(it:it+9,1));
        % pi(n) = P(n-1)*u(n)
        pi_n = P*u_n;
        % Gain Vector: k(n) = (lamda + u^H(n)*pi(n))^-1 * pi(n)
        k_n = inv(lamda + u_n'*pi_n)*pi_n;
        % A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        ktzi_n = d(it+9,1) - w(:,it)'*u_n;
        % w(n) = w(n-1) + k(n)*conj(ktzi(n))
        w(:,it+1) = w(:,it) + k_n*conj(ktzi_n);
        % Prepare for next iteration: P(n) = inv(lamda)*P(n-1) - inv(lamda)*k(n)*pi^H(n)
        P = inv(lamda)*P - inv(lamda)*k_n*pi_n';

        % Learning Curves
        % Before Update/A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        % J'(n) = E(|ktzi(n)|^2)
        J_ktzi_C1(MonteCarlo,it) = abs(ktzi_n).^2;
        % After Update/A-Posteriori Estimation Error: e(n) = d(n) - w^H(n)*u(n)
        % J(n) = E(|e(n)|^2)
        e_n = d(it+9,1) - w(:,it+1)'*u_n;
        J_e_C1(MonteCarlo,it) = abs(e_n).^2;
    end 
end

% Case 2
J_ktzi_C2 = zeros(100,num_iters);
J_e_C2 = zeros(100,num_iters);

for MonteCarlo = 1:100 %run the experiment 100 times so that can average Js
    
    num_iters = 100; %tune until algo converges and P(0) negligable

    [u,d] = u_and_d(num_iters+9,2); %generate u[n] and d[n] data
    w = zeros(10,num_iters+1); %initialize w(0) = 0, index off by one bc w(0) in first location
    P = (delta^-1) .* eye(10); %initialize P(0) = inv(delta)*I, i.e. phi(0) = del*I

    for it = 1:num_iters

        % RLS Equations
        % u_10(n) = [u(n) u(n-1) ... u(n-9)]'
        u_n = flipud(u(it:it+9,1));
        % pi(n) = P(n-1)*u(n)
        pi_n = P*u_n;
        % Gain Vector: k(n) = (lamda + u^H(n)*pi(n))^-1 * pi(n)
        k_n = inv(lamda + u_n'*pi_n)*pi_n;
        % A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        ktzi_n = d(it+9,1) - w(:,it)'*u_n;
        % w(n) = w(n-1) + k(n)*conj(ktzi(n))
        w(:,it+1) = w(:,it) + k_n*conj(ktzi_n);
        % Prepare for next iteration: P(n) = inv(lamda)*P(n-1) - inv(lamda)*k(n)*pi^H(n)
        P = inv(lamda)*P - inv(lamda)*k_n*pi_n';

        % Learning Curves
        % Before Update/A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        % J'(n) = E(|ktzi(n)|^2)
        J_ktzi_C2(MonteCarlo,it) = abs(ktzi_n).^2;
        % After Update/A-Posteriori Estimation Error: e(n) = d(n) - w^H(n)*u(n)
        % J(n) = E(|e(n)|^2)
        e_n = d(it+9,1) - w(:,it+1)'*u_n;
        J_e_C2(MonteCarlo,it) = abs(e_n).^2;
    end 
end

% Graph RLS Results
figure
sgtitle('RLS M=10 Too Many Tap Weights: Case 1 = poles 0.3 and 0.5, Case 2 = poles 0.3 and 0.95')
subplot(2,2,2)
plot(1:num_iters,J_e_C1(1,:))
hold on
plot(1:num_iters,J_ktzi_C1(1,:))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('J Learning Curves for One Simulation of RLS Case 1')
xlabel('adaptive iteration number')

subplot(2,2,1)
plot(1:num_iters,mean(J_e_C1))
hold on
plot(1:num_iters,mean(J_ktzi_C1))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('Average J Learning Curves for 100 Simulations of RLS Case 1')
xlabel('adaptive iteration number')

subplot(2,2,4)
plot(1:num_iters,J_e_C2(1,:))
hold on
plot(1:num_iters,J_ktzi_C2(1,:))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('J Learning Curves for One Simulation of RLS Case 2')
xlabel('adaptive iteration number')

subplot(2,2,3)
plot(1:num_iters,mean(J_e_C2))
hold on
plot(1:num_iters,mean(J_ktzi_C2))
legend('A-Posteriori Learning Curve = E( |e(n)|.^2 )','A-Priori Learning Curve = E( |ktzi(n)|.^2 )')
title('Average J Learning Curves for 100 Simulations of RLS Case 2')
xlabel('adaptive iteration number')

% Performs very well, despite having too few tap weights because RLS is
% much better than LMS. Takes longer to converge --> extra tap weights
% allows for more "sloshing around."

%% Adaptive Equalizer RLS

% Perform 100 independent Monte Carlo runs to generate reasonable learning curves

lamda = 0.95;
delta = 0.005;

J_e_i = zeros(100,989);
J_e_ii = zeros(100,989);
J_e_iii = zeros(100,989);

J_ktzi_i = zeros(100,989);
J_ktzi_ii = zeros(100,989);
J_ktzi_iii = zeros(100,989);

mu = 0.05;  

for runs = 1:100
    
    % Channel Input x_n - Bernoulli sequence with x_n = +1/-1
    x = randi(2,1000,1)-1;
    x(x==0) = -1;
    % Additive White Gaussian Noise with zero mean and variance = 0.01
    v = 0.01./sqrt(2).*(randn(998,1));

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

    P_i = (delta^-1) .* eye(21); %initialize P(0) = inv(delta)*I, i.e. phi(0) = del*I
    P_ii = P_i;
    P_iii = P_i;
    
    for n = 11:988

        % RLS Equations
        % u_21(n) = [u(n+10),u(n+9),...,u(n),...,u(n-10)]'
        u_n_i = flipud(u_i(n-10:n+10,1));
        u_n_ii = flipud(u_ii(n-10:n+10,1));
        u_n_iii = flipud(u_iii(n-10:n+10,1));
        
        % pi(n) = P(n-1)*u(n)
        pi_n_i = P_i*u_n_i;
        pi_n_ii = P_ii*u_n_ii;
        pi_n_iii = P_iii*u_n_iii;
        
        % Gain Vector: k(n) = (lamda + u^H(n)*pi(n))^-1 * pi(n)
        k_n_i = inv(lamda + u_n_i'*pi_n_i)*pi_n_i;
        k_n_ii = inv(lamda + u_n_ii'*pi_n_ii)*pi_n_ii;
        k_n_iii = inv(lamda + u_n_iii'*pi_n_iii)*pi_n_iii;
        
        % A-Priori Estimation Error: ktzi(n) = d(n)/x(n) - w^H(n-1)*u(n)
        ktzi_n_i = x(n,1) - taps_i(:,runs)'*u_n_i;
        ktzi_n_ii = x(n,1) - taps_ii(:,runs)'*u_n_ii;
        ktzi_n_iii = x(n,1) - taps_iii(:,runs)'*u_n_iii;
        
        % w(n) = w(n-1) + k(n)*conj(ktzi(n))
        taps_i(:,runs) = taps_i(:,runs) + k_n_i*conj(ktzi_n_i);
        taps_ii(:,runs) = taps_ii(:,runs) + k_n_ii*conj(ktzi_n_ii);
        taps_iii(:,runs) = taps_iii(:,runs) + k_n_iii*conj(ktzi_n_iii);
        
        % e(n) = d(n)/x(n) - w^H(n)*u(n)
        e_n_i = x(n,1) - taps_i(:,runs)'*u_n_i;
        e_n_ii = x(n,1) - taps_ii(:,runs)'*u_n_ii;
        e_n_iii = x(n,1) - taps_iii(:,runs)'*u_n_iii;
        
        % Prepare for next iteration: P(n) = inv(lamda)*P(n-1) - inv(lamda)*k(n)*pi^H(n)
        P_i = inv(lamda)*P_i - inv(lamda)*k_n_i*pi_n_i';
        P_ii = inv(lamda)*P_ii - inv(lamda)*k_n_ii*pi_n_ii';
        P_iii = inv(lamda)*P_iii - inv(lamda)*k_n_iii*pi_n_iii';

        % Learning Curves
        % Before Update/A-Priori Estimation Error: ktzi(n) = d(n) - w^H(n-1)*u(n)
        % J'(n) = E(|ktzi(n)|^2)
        J_ktzi_i(runs,n-9) = abs(ktzi_n_i).^2;
        J_ktzi_ii(runs,n-9) = abs(ktzi_n_ii).^2;
        J_ktzi_iii(runs,n-9) = abs(ktzi_n_iii).^2;
        % After Update/A-Posteriori Estimation Error: e(n) = d(n) - w^H(n)*u(n)
        % J(n) = E(|e(n)|^2)
        J_e_i(runs,n-9) = abs(e_n_i).^2;
        J_e_ii(runs,n-9) = abs(e_n_ii).^2;
        J_e_iii(runs,n-9) = abs(e_n_iii).^2;

    end
end   

% Average J values over Monte Carlo experiments
avgJe_i = mean(J_e_i);
avgJe_ii = mean(J_e_ii);
avgJe_iii = mean(J_e_iii);

avgJk_i = mean(J_ktzi_i);
avgJk_ii = mean(J_ktzi_ii);
avgJk_iii = mean(J_ktzi_iii);
%%
% Graph RLS Results
figure
subplot(2,1,1)
plot(1:50,avgJk_i(1:50))
hold on
plot(1:50,avgJk_ii(1:50))
hold on
plot(1:50,avgJk_iii(1:50))
legend('Ji k','Jii k','Jiii k')
xlabel('n (iteration of adaptive algorithm)')
title('RLS Adaptive Equalizer A-Posteriori Learning Curves J = E( |e(n)|.^2 )')

subplot(2,1,2)
plot(1:50,avgJe_i(1:50))
hold on
plot(1:50,avgJe_ii(1:50))
hold on
plot(1:50,avgJe_iii(1:50))
hold on
legend('Ji e','Jii e','Jiii e')
xlabel('n (iteration of adaptive algorithm)')
title('RLS Adaptive Equalizer A-Priori Learning Curves J = E( |ktzi(n)|.^2 )')

% As expected, the a-priori learning curve values for small n are extremely
% large but drop off quickly. The a-posteriori learning curve values for
% small n are very small but grow a little more. Overall, both curves
% converge extremely quickly. When n<=M, the a-postreiori error is exactly
% zero as expected from RLS.



%% 3. QRD-RLS

% Below test all of the function that were built at end of script

[u,d] = u_and_d(num_iters+5,1);
lamda = 0.95;
u_n = flipud(u(1:6,1));
d_n = d(6,1);

% Test the QRD-RLS core function
phi_init = ((0.005)^0.5).*eye(6);
p_init = repmat(0,6,1)';
% phi^1/2 is 6-by-6 and p is 6-by-1, M=6
[e_n,sqrt_phi_n,p_n] = QRD_RLS(u_n,d_n,lamda,phi_init,p_init);

% Test the inv QRD-RLS core function
P_init = ((0.005)^-0.5).*eye(6);
w_init = zeros(6,1);
[w_n,sqrt_P_n,ktzi,sqrt_gamma] = inv_QRD_RLS(u_n,d_n,lamda,P_init,w_init);

% Test the general QRD-RLS function
[e,sqrt_phi_n,p_H_n] = gen_QRD_RLS(u,d,6,lamda,0.005);
% Test the general inv QRD-RLS function
[w,ktzi,sqrt_gamma] = gen_inv_QRD_RLS(u,d,6,lamda,0.005);

% Test the time series QRD_RLS function
[e,sqrt_phi,p] = ts_QRD_RLS(u,d,6,lamda,0.005,inf);
% Test the time series inv QRD_RLS function
[w,ktzi,sqrt_gamma] = ts_inv_QRD_RLS(u,d,6,lamda,delta,inf);

% Test the function that computes w for QRD_RLS
w_n = QRD_RLS_w(p_H_n,sqrt_phi);
% Test the function that computes e for QRD_RLS
e_n = inv_QRD_RLS_e(ktzi,sqrt_gamma);


%% Run adaptive equalizer QRD-RLS and inverse QRD-RLS
lamda = 0.95;
delta = 0.005;
mu = 0.05;  
   
% Channel Input x_n - Bernoulli sequence with x_n = +1/-1
x = randi(2,1000,1)-1;
x(x==0) = -1;
% Additive White Gaussian Noise with zero mean and variance = 0.01
v = 0.01./sqrt(2).*(randn(998,1));

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

% Build an adaptive equalizer consisting of an FIR filter with 21 taps
taps_i = zeros(21,1);
taps_ii = zeros(21,1);
taps_iii = zeros(21,1);

% All of the channel transfer functions indicate that the channel preserves
% the input, delays it by one, and adds a portion of the symbol before and
% after.
% Take delta = 10, meaning taps 0-9 represent future symbols that enter
% reciever after target point and taps 11-20 represent past symbols that
% entered before target point. 
% To initialize, set the center tap to 1 and all else to 0.
taps_i(11,1) = 1;
taps_ii(11,1) = 1;
taps_iii(11,1) = 1;

% In order for the x/d to properly line up, because examining the
% middle tap of index 11, must add dummy 10 values to beginning to shift all.
x = [zeros(10,1);x];

[e_i,sqrt_phi_i,P_i] = gen_QRD_RLS(u_i,x,21,lamda,delta);
[e_ii,sqrt_phi_ii,P_ii] = gen_QRD_RLS(u_ii,x,21,lamda,delta);
[e_iii,sqrt_phi_iii,P_iii] = gen_QRD_RLS(u_iii,x,21,lamda,delta);

w_i_reg = QRD_RLS_w(P_i,sqrt_phi_i);
w_ii_reg = QRD_RLS_w(P_ii,sqrt_phi_ii);
w_iii_reg = QRD_RLS_w(P_iii,sqrt_phi_iii);

[w_i_inv,ktzi_i,sqrt_gamma_i] = gen_inv_QRD_RLS(u_i,x,21,lamda,delta,taps_i);
[w_ii_inv,ktzi_ii,sqrt_gamma_ii] = gen_inv_QRD_RLS(u_ii,x,21,lamda,delta,taps_ii);
[w_iii_inv,ktzi_iii,sqrt_gamma_iii] = gen_inv_QRD_RLS(u_iii,x,21,lamda,delta,taps_iii);

e_i_inv = inv_QRD_RLS_e(ktzi_i,sqrt_gamma_i);
e_ii_inv = inv_QRD_RLS_e(ktzi_ii,sqrt_gamma_ii);
e_iii_inv = inv_QRD_RLS_e(ktzi_iii,sqrt_gamma_iii);


% W doesnt match exactly

%% BER

% Generate 10,000 data points and run them through the decoder. Compute BER
% Channel Input x_n 10,000 Bernoulli samples 
x = randi(2,10022,1)-1;
x(x==0) = -1;
% Additive White Gaussian Noise with zero mean and variance = 0.01
v = 0.01./sqrt(2).*(randn(10020,1));
% Shift x by 1 = x[n-1]
x_shift1 = circshift(x,1);
% Shift x by 2 = x[n-2]
x_shift2 = circshift(x,2);
% Remove first 2 meaningless values
x = x(3:end,:);
x_shift1 = x_shift1(3:end,:);
x_shift2 = x_shift2(3:end,:);

u_i = 0.25.*x + x_shift1 + 0.25.*x_shift2 + v;
u_ii = 0.25.*x + x_shift1 - 0.25.*x_shift2 + v;
u_iii = -0.25.*x + x_shift1 + 0.25.*x_shift2 + v;

decode = zeros(10000,1);
orig = zeros(10000,1);
for j=1:10020-20
    u_i_filt = flipud(u_i(j:j+20,1));
    u_ii_filt = flipud(u_ii(j:j+20,1));
    u_iii_filt = flipud(u_iii(j:j+20,1));
    
    orig(j,1) = x(j+10,1);
    
    decode_i(j,1) = w_i_reg'*u_i_filt;
    decode_ii(j,1) = w_ii_reg'*u_ii_filt;
    decode_iii(j,1) = w_iii_reg'*u_iii_filt;
end

% Use a simple decoder - if pos, it is a 1. if neg, it is a -1
decode_decision_i = decode_i;
decode_decision_i(decode_decision_i>0) = 1;
decode_decision_i(decode_decision_i<0) = -1;
BER_i = sum(orig-decode_decision_i) %ALL were determined correctly!!

decode_decision_ii = decode_ii;
decode_decision_ii(decode_decision_ii>0) = 1;
decode_decision_ii(decode_decision_ii<0) = -1;
BER_ii = sum(orig-decode_decision_ii) %ALL were determined correctly!!

decode_decision_iii = decode_iii;
decode_decision_iii(decode_decision_iii>0) = 1;
decode_decision_iii(decode_decision_iii<0) = -1;
BER_iii = sum(orig-decode_decision_iii) %ALL were determined correctly!!

%% 5. QRD-LSL

% Poles of AR[3] model
k = [0,1,2];
poles = 0.95*exp(sqrt(-1)*2*pi*k./3);

% Convert into difference equation
% H = 1./z^3*(1-p1*z^-1)*(1-p2*z^-1)*(1-p3*z^-1)
p1 = [1 -poles(1)];
p2 = [1 -poles(2)];
p3 = [1 -poles(3)];

p1p2 = conv(p1,p2);
denom = conv(p1p2,p3);

[b,a]=zp2tf([],poles,1); %yields same as above method

a1 = denom(2);
a2 = denom(3);
a3 = denom(4);

% u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]

% Unit variance, complex white noise signal v of length 10^3 
v = (1/sqrt(2))*(randn(10^3,1));
u = zeros(10^3,1);
d = zeros(10^3,1);

% Initialize
u(1,:) = v(1,:);
u(2,:) = v(2,:) - a1*u(1,:);
u(3,:) = v(3,:) - a1*u(2,:) - a2*u(3,:);

d(1,:) = v(1,:) + u(1,:);
d(2,:) = v(2,:) + u(2,:) + u(1,:);
d(3,:) = v(3,:) + u(3,:) + u(2,:) + u(1,:);

% Compute u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]
% and d[n] = v[n] + u[n] + u[n-1] + u[n-2]
for i=4:10^3
    u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
    d(i,:) = v(i,:) + u(i,:) + u(i-1,:) + u(i-2,:);   
end 

%% (A) 
% Using time averages, find correlation matrix R for u_M(n)
corr_coeffs = zeros(9,55);
for i=1:10^3 -2
   u_M = flipud(u(i:i+2,:));
   R = u_M*u_M';
   corr_coeffs(:,i) = R(:);
end

corr = mean(corr_coeffs,2);
R = [corr(1,1) corr(4,1) corr(7,1); corr(2,1) corr(5,1) corr(8,1); corr(3,1) corr(6,1) corr(9,1)]
    
% Select reasonable mu for the LMS algorithm
% mu_max = 2./max(eig(R))
mu_max = 2./max(eig(R))
mu = 0.05*mu_max

w0 = [1;1;1];

%% (B)
% The formula for LMS is w(n+1) = w(n) + mu*u_M(n)*conj(e(n)) where
% conj(e(n)) = (d(n)-w^H(n)*u_M(n))^H)

J = zeros(100,100);
D = zeros(100,100);
    
for runs = 1:100
    % Generate fresh u and d stochastic signals for each run
    
    % Unit variance, complex white noise signal v of length 10^3 
    v = (1/sqrt(2))*(randn(103,1));
    u = zeros(103,1);
    d = zeros(103,1);
    % Initialize
    u(1,:) = v(1,:);
    u(2,:) = v(2,:) - a1*u(1,:);
    u(3,:) = v(3,:) - a1*u(2,:) - a2*u(3,:);
    d(1,:) = v(1,:) + u(1,:);
    d(2,:) = v(2,:) + u(2,:) + u(1,:);
    d(3,:) = v(3,:) + u(3,:) + u(2,:) + u(1,:);
    % Compute u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]
    % and d[n] = v[n] + u[n] + u[n-1] + u[n-2]
    for i=4:103
        u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
        d(i,:) = v(i,:) + u(i,:) + u(i-1,:) + u(i-2,:);   
    end
    
    % Initialize w
    w = zeros(3,1);   
    for itr = 1:97

        % y = w^H*u
        y = w'*flipud(u(itr:itr+2,1));

        % err = d - y
        err = (d(itr+3,1) - y)';

        % w(n+1) = w(n) + mu*u_M(n)*conj(e(n))
        w = w + mu*flipud(u(itr:itr+2,1))*err;

        % MSE Learning Curve J(n) = E(|e(n)|^2)
        J(runs,itr) = abs(d(itr+2,1) - y).^2;
        % Mean Square Deviation Learning Curve D(n) = E(||w0-what||^2)
        D(runs,itr) = norm(w0-w).^2;

    end
end

J = mean(J);
D = mean(D);

figure
plot(J)
hold on
plot(D)
title('Q5. Learning Curves for LMS algorithm to estimate w0')
legend('J','D')
xlabel('iteration n')

%% (C) QRD-LSL
% Data
N=50;
% Generate fresh u and d stochastic signals for each run
    % Unit variance, complex white noise signal v of length 10^3 
    v = (1/sqrt(2))*(randn(N+3,1));
    u = zeros(N+3,1);
    d = zeros(N+3,1);
    % Initialize
    u(1,:) = v(1,:);
    u(2,:) = v(2,:) - a1*u(1,:);
    u(3,:) = v(3,:) - a1*u(2,:) - a2*u(3,:);
    d(1,:) = v(1,:) + u(1,:);
    d(2,:) = v(2,:) + u(2,:) + u(1,:);
    d(3,:) = v(3,:) + u(3,:) + u(2,:) + u(1,:);
    % Compute u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]
    % and d[n] = v[n] + u[n] + u[n-1] + u[n-2]
    for i=4:N+3
        u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
        d(i,:) = v(i,:) + u(i,:) + u(i-1,:) + u(i-2,:);   
    end
    
    
lamda = 1;
delta = 0.01;

% Number of Timesteps
N = 50;
% Order M goes from 0 to 3 

% Storage variables

% Starts at n=-1, followed by n=0
curlyB = zeros(N+2,4); %time-by-orders
% Starts at n=0
curlyF = zeros(N+1,4); %time-by-orders
% Starts at n=0
gam = zeros(N+1,4);

% All e's start at n=1
ef = zeros(N,4);
eb = zeros(N,4);
em = zeros(N,4);

% Starts at n=0
pf = zeros(N+1,4);
% Starts at n=0
pb = zeros(N+1,4);
% Starts at n=0, M goes up to 4
pm = zeros(N,5);


% Initialization Stage
%   (a) Auxiliary parameter initialization. For order m=0,1,2 set
% pf,m(0) = pb,m(0) = 0 (already done) and pm(0) = 0 [includes order 3]
%   (b) Soft-constraint initialization. For order m=0,1,2,3 set
% curlyBm(-1) = curlyBm(0) = delta
curlyB(1:2,:) = delta^0.5;
% curlyFm(0) = delta
curlyF(1,:) = delta^0.5;
%   (c) Data initialization. For n=1,2,... compute
% ef,m=0(n) = eb,m=0(n) = u(n)
ef(:,1) = u(1:N,:); % M=0 for all time
eb(:,1) = u(1:N,:); % M=0 for all time
% e,m=0(n) = d(n)
em(:,1) = d(1:N,:); % M=0 for all time
% gam,m=0(n) = 1
gam(:,1) = 1^0.5;

%%
% Run QRD-LSL

for m=2:4
    for n=2:N
        [curlyB(n-1+1,m-1),ef(n,m),pf(n,m-1),gam(n-1,m)] = Annil_b_m1_n1(lamda,curlyB(n-2+1,m-1),eb(n-1,m-1),ef(n,m-1),pf(n-1,m-1),gam(n-1,m-1));
        [curlyF(n,m-1),eb(n,m),pb(n,m-1)] = Annil_f_m1_n(lamda,curlyF(n-1,m-1),ef(n,m-1),eb(n-1,m-1),pb(n-1,m-1));
        [curlyB(n+1,m-1),em(n,m),pm(n,m-1)] = Annil_b_m1_n(lamda,curlyB(n-1+1,m-1),eb(n,m-1),em(n,m-1),pm(n-1,m-1));
    end
end

for n=2:N
    [curlyB(n-1+1,4),em(n,5),pm(n,5)] = Annil_b_m1_n(lamda,curlyB(n-2+1,4),eb(n,4),em(n,4),pm(n-1,4));
end


%% Learning Curve J = E(|d-y|^2)
% Here, y(n) = conj(epsilon_m1_n). This is the value to be stored that we
% care about so that the learning curve can be plotted.
%J = abs(em(:,4).*gam(2:N+1,4)).^2;
%figure
%plot(J)
%title('J =  E(|e|^2) = E(|(em*gamma)|^2')

% To find h regression coefficients: 
% h_m-1(n) = p_m-1(n)./curlyB_m-1(n)
h_1 = pm(:,1)./curlyB(3:end,1);
h_2 = pm(:,2)./curlyB(3:end,2);
h_3 = pm(:,3)./curlyB(3:end,3);
h_4 = pm(:,4)./curlyB(3:end,4);

w_hat = [h_1 h_2 h_3 h_4]';
learning = vecnorm(repmat([1;1;1;1],1,N) - w_hat).^2;
figure
plot(learning)
title('Learning Curve for the QRD-LSL')


%% Perform above over 100 simulations (Monte Carlo)
D_mc=zeros(100,50);

for runs=1:100
    % Data
    N=50;
    % Generate fresh u and d stochastic signals for each run
        % Unit variance, complex white noise signal v of length 10^3 
        v = (1/sqrt(2))*(randn(N+3,1));
        u = zeros(N+3,1);
        d = zeros(N+3,1);
        % Initialize
        u(1,:) = v(1,:);
        u(2,:) = v(2,:) - a1*u(1,:);
        u(3,:) = v(3,:) - a1*u(2,:) - a2*u(3,:);
        d(1,:) = v(1,:) + u(1,:);
        d(2,:) = v(2,:) + u(2,:) + u(1,:);
        d(3,:) = v(3,:) + u(3,:) + u(2,:) + u(1,:);
        % Compute u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]
        % and d[n] = v[n] + u[n] + u[n-1] + u[n-2]
        for i=4:N+3
            u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
            d(i,:) = v(i,:) + u(i,:) + u(i-1,:) + u(i-2,:);   
        end


    lamda = 1;
    delta = 0.0001;

    % Order M goes from 0 to 3 

    % Storage variables

    % Starts at n=-1, followed by n=0
    curlyB = zeros(N+2,4); %time-by-orders
    % Starts at n=0
    curlyF = zeros(N+1,4); %time-by-orders
    % Starts at n=0
    gam = zeros(N+1,4);

    % All e's start at n=1
    ef = zeros(N,4);
    eb = zeros(N,4);
    em = zeros(N,4);

    % Starts at n=0
    pf = zeros(N+1,4);
    % Starts at n=0
    pb = zeros(N+1,4);
    % Starts at n=0, M goes up to 4
    pm = zeros(N,5);


    % Initialization Stage
    %   (a) Auxiliary parameter initialization. For order m=0,1,2 set
    % pf,m(0) = pb,m(0) = 0 (already done) and pm(0) = 0 [includes order 3]
    %   (b) Soft-constraint initialization. For order m=0,1,2,3 set
    % curlyBm(-1) = curlyBm(0) = delta
    curlyB(1:2,:) = delta^0.5;
    % curlyFm(0) = delta
    curlyF(1,:) = delta^0.5;
    %   (c) Data initialization. For n=1,2,... compute
    % ef,m=0(n) = eb,m=0(n) = u(n)
    ef(:,1) = u(1:N,:); % M=0 for all time
    eb(:,1) = u(1:N,:); % M=0 for all time
    % e,m=0(n) = d(n)
    em(:,1) = d(1:N,:); % M=0 for all time
    % gam,m=0(n) = 1
    gam(:,1) = 1^0.5;

    % Run QRD-LSL

    for m=2:4
        for n=2:N
            [curlyB(n-1+1,m-1),ef(n,m),pf(n,m-1),gam(n-1,m)] = Annil_b_m1_n1(lamda,curlyB(n-2+1,m-1),eb(n-1,m-1),ef(n,m-1),pf(n-1,m-1),gam(n-1,m-1));
            [curlyF(n,m-1),eb(n,m),pb(n,m-1)] = Annil_f_m1_n(lamda,curlyF(n-1,m-1),ef(n,m-1),eb(n-1,m-1),pb(n-1,m-1));
            [curlyB(n+1,m-1),em(n,m),pm(n,m-1)] = Annil_b_m1_n(lamda,curlyB(n-1+1,m-1),eb(n,m-1),em(n,m-1),pm(n-1,m-1));
        end
    end

    for n=2:N
        [curlyB(n-1+1,4),em(n,5),p(n,4)] = Annil_b_m1_n(lamda,curlyB(n-2+1,4),eb(n,4),em(n,4),pm(n-1,5));
    end

    % To find h regression coefficients: 
    % h_m-1(n) = p_m-1(n)./curlyB_m-1(n)
    h_1 = pm(:,1)./curlyB(3:end,1);
    h_2 = pm(:,2)./curlyB(3:end,2);
    h_3 = pm(:,3)./curlyB(3:end,3);
    h_4 = pm(:,4)./curlyB(3:end,4);

    w_hat = [h_1 h_2 h_3 h_4]';
    D_mc(runs,:) = vecnorm(repmat([1;1;1;1],1,N) - w_hat).^2;

end

figure
plot(mean(D_mc))
title('Learning Curve for the QRD-LSL averaged over 100 simulations (D)')


%% (6) Nonstationarity

% It takes approximately 20-25 iterations for the filter to converge
conv = 25; 

% Nonstationarity every N/2
D_mc_Nov2=zeros(100,50);

for runs=1:100
    % Data
    N=50;
    % Generate fresh u and d stochastic signals for each run
        % Unit variance, complex white noise signal v 
        v = (1/sqrt(2))*(randn(N+3,1));
        u = zeros(N+3,1);
        d = zeros(N+3,1);
        % Initialize
        u(1,:) = v(1,:);
        u(2,:) = v(2,:) - a1*u(1,:);
        u(3,:) = v(3,:) - a1*u(2,:) - a2*u(3,:);
        d(1,:) = v(1,:) + u(1,:);
        d(2,:) = v(2,:) + u(2,:) + u(1,:);
        d(3,:) = v(3,:) + u(3,:) + u(2,:) + u(1,:);
       
        opt_w_Nov2 = zeros(4,N);
        
        % Compute u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]
        % and d[n] = v[n] + u[n] + u[n-1] + u[n-2]
        for i=[4:12 25:37]
            u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
            d(i,:) = v(i,:) + u(i,:) + u(i-1,:) + u(i-2,:); 
            opt_w_Nov2(:,i) = [1;1;1;1];
        end
        
        % Every N/2 switch to [-1 -1 -1 -1] as w0
        for i=[13:24 38:50]
            u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
            d(i,:) = v(i,:) - u(i,:) - u(i-1,:) - u(i-2,:);  
            opt_w_Nov2(:,i) = [-1;-1;-1;-1];
        end

    lamda = 1;
    delta = 0.0001;

    % Number of Timesteps
    N = 50;
    % Order M goes from 0 to 3 

    % Storage variables

    % Starts at n=-1, followed by n=0
    curlyB = zeros(N+2,4); %time-by-orders
    % Starts at n=0
    curlyF = zeros(N+1,4); %time-by-orders
    % Starts at n=0
    gam = zeros(N+1,4);

    % All e's start at n=1
    ef = zeros(N,4);
    eb = zeros(N,4);
    em = zeros(N,4);

    % Starts at n=0
    pf = zeros(N+1,4);
    % Starts at n=0
    pb = zeros(N+1,4);
    % Starts at n=0, M goes up to 4
    pm = zeros(N,5);


    % Initialization Stage
    %   (a) Auxiliary parameter initialization. For order m=0,1,2 set
    % pf,m(0) = pb,m(0) = 0 (already done) and pm(0) = 0 [includes order 3]
    %   (b) Soft-constraint initialization. For order m=0,1,2,3 set
    % curlyBm(-1) = curlyBm(0) = delta
    curlyB(1:2,:) = delta^0.5;
    % curlyFm(0) = delta
    curlyF(1,:) = delta^0.5;
    %   (c) Data initialization. For n=1,2,... compute
    % ef,m=0(n) = eb,m=0(n) = u(n)
    ef(:,1) = u(1:N,:); % M=0 for all time
    eb(:,1) = u(1:N,:); % M=0 for all time
    % e,m=0(n) = d(n)
    em(:,1) = d(1:N,:); % M=0 for all time
    % gam,m=0(n) = 1
    gam(:,1) = 1^0.5;

    % Run QRD-LSL

    for m=2:4
        for n=2:N
            [curlyB(n-1+1,m-1),ef(n,m),pf(n,m-1),gam(n-1,m)] = Annil_b_m1_n1(lamda,curlyB(n-2+1,m-1),eb(n-1,m-1),ef(n,m-1),pf(n-1,m-1),gam(n-1,m-1));
            [curlyF(n,m-1),eb(n,m),pb(n,m-1)] = Annil_f_m1_n(lamda,curlyF(n-1,m-1),ef(n,m-1),eb(n-1,m-1),pb(n-1,m-1));
            [curlyB(n+1,m-1),em(n,m),pm(n,m-1)] = Annil_b_m1_n(lamda,curlyB(n-1+1,m-1),eb(n,m-1),em(n,m-1),pm(n-1,m-1));
        end
    end

    for n=2:N
        [curlyB(n-1+1,4),em(n,5),p(n,4)] = Annil_b_m1_n(lamda,curlyB(n-2+1,4),eb(n,4),em(n,4),pm(n-1,5));
    end

    % To find h regression coefficients: 
    % h_m-1(n) = p_m-1(n)./curlyB_m-1(n)
    h_1 = pm(:,1)./curlyB(3:end,1);
    h_2 = pm(:,2)./curlyB(3:end,2);
    h_3 = pm(:,3)./curlyB(3:end,3);
    h_4 = pm(:,4)./curlyB(3:end,4);

    w_hat = [h_1 h_2 h_3 h_4]';
    D_mc_Nov2(runs,:) = vecnorm(opt_w_Nov2 - w_hat).^2;

end


% Nonstationarity every N
D_mc_N=zeros(100,100);

for runs=1:100
    % Data
    N=100;
    % Generate fresh u and d stochastic signals for each run
        % Unit variance, complex white noise signal v 
        v = (1/sqrt(2))*(randn(N+3,1));
        u = zeros(N+3,1);
        d = zeros(N+3,1);
        % Initialize
        u(1,:) = v(1,:);
        u(2,:) = v(2,:) - a1*u(1,:);
        u(3,:) = v(3,:) - a1*u(2,:) - a2*u(3,:);
        d(1,:) = v(1,:) + u(1,:);
        d(2,:) = v(2,:) + u(2,:) + u(1,:);
        d(3,:) = v(3,:) + u(3,:) + u(2,:) + u(1,:);
       
        opt_w_N = zeros(4,N);
        
        % Compute u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]
        % and d[n] = v[n] + u[n] + u[n-1] + u[n-2]
        for i=[4:24 51:74]
            u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
            d(i,:) = v(i,:) + u(i,:) + u(i-1,:) + u(i-2,:); 
            opt_w_N(:,i) = [1;1;1;1];
        end
        
        % Every N/2 switch to [-1 -1 -1 -1] as w0
        for i=[25:50 75:100]
            u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
            d(i,:) = v(i,:) - u(i,:) - u(i-1,:) - u(i-2,:);  
            opt_w_Nov2(:,i) = [-1;-1;-1;-1];
        end

    lamda = 1;
    delta = 0.0001;
    
    % Storage variables

    % Starts at n=-1, followed by n=0
    curlyB = zeros(N+2,4); %time-by-orders
    % Starts at n=0
    curlyF = zeros(N+1,4); %time-by-orders
    % Starts at n=0
    gam = zeros(N+1,4);

    % All e's start at n=1
    ef = zeros(N,4);
    eb = zeros(N,4);
    em = zeros(N,4);

    % Starts at n=0
    pf = zeros(N+1,4);
    % Starts at n=0
    pb = zeros(N+1,4);
    % Starts at n=0, M goes up to 4
    pm = zeros(N,5);


    % Initialization Stage
    %   (a) Auxiliary parameter initialization. For order m=0,1,2 set
    % pf,m(0) = pb,m(0) = 0 (already done) and pm(0) = 0 [includes order 3]
    %   (b) Soft-constraint initialization. For order m=0,1,2,3 set
    % curlyBm(-1) = curlyBm(0) = delta
    curlyB(1:2,:) = delta^0.5;
    % curlyFm(0) = delta
    curlyF(1,:) = delta^0.5;
    %   (c) Data initialization. For n=1,2,... compute
    % ef,m=0(n) = eb,m=0(n) = u(n)
    ef(:,1) = u(1:N,:); % M=0 for all time
    eb(:,1) = u(1:N,:); % M=0 for all time
    % e,m=0(n) = d(n)
    em(:,1) = d(1:N,:); % M=0 for all time
    % gam,m=0(n) = 1
    gam(:,1) = 1^0.5;

    % Run QRD-LSL

    for m=2:4
        for n=2:N
            [curlyB(n-1+1,m-1),ef(n,m),pf(n,m-1),gam(n-1,m)] = Annil_b_m1_n1(lamda,curlyB(n-2+1,m-1),eb(n-1,m-1),ef(n,m-1),pf(n-1,m-1),gam(n-1,m-1));
            [curlyF(n,m-1),eb(n,m),pb(n,m-1)] = Annil_f_m1_n(lamda,curlyF(n-1,m-1),ef(n,m-1),eb(n-1,m-1),pb(n-1,m-1));
            [curlyB(n+1,m-1),em(n,m),pm(n,m-1)] = Annil_b_m1_n(lamda,curlyB(n-1+1,m-1),eb(n,m-1),em(n,m-1),pm(n-1,m-1));
        end
    end

    for n=2:N
        [curlyB(n-1+1,4),em(n,5),p(n,4)] = Annil_b_m1_n(lamda,curlyB(n-2+1,4),eb(n,4),em(n,4),pm(n-1,5));
    end

    % To find h regression coefficients: 
    % h_m-1(n) = p_m-1(n)./curlyB_m-1(n)
    h_1 = pm(:,1)./curlyB(3:end,1);
    h_2 = pm(:,2)./curlyB(3:end,2);
    h_3 = pm(:,3)./curlyB(3:end,3);
    h_4 = pm(:,4)./curlyB(3:end,4);

    w_hat = [h_1 h_2 h_3 h_4]';
    D_mc_N(runs,:) = vecnorm(opt_w_N - w_hat).^2;

end

% It takes approximately 20-25 iterations for the filter to converge
conv = 25; 

% Nonstationarity every 2*N
D_mc_2N=zeros(100,100);

for runs=1:100
    % Data
    N=100;
    % Generate fresh u and d stochastic signals for each run
        % Unit variance, complex white noise signal v 
        v = (1/sqrt(2))*(randn(N+3,1));
        u = zeros(N+3,1);
        d = zeros(N+3,1);
        % Initialize
        u(1,:) = v(1,:);
        u(2,:) = v(2,:) - a1*u(1,:);
        u(3,:) = v(3,:) - a1*u(2,:) - a2*u(3,:);
        d(1,:) = v(1,:) + u(1,:);
        d(2,:) = v(2,:) + u(2,:) + u(1,:);
        d(3,:) = v(3,:) + u(3,:) + u(2,:) + u(1,:);
       
        opt_w_Nov2 = zeros(4,N);
        
        % Compute u[n] = v[n] - a1*u[n-1] - a2*u[n-2] - a3*u[n-3]
        % and d[n] = v[n] + u[n] + u[n-1] + u[n-2]
        for i=[4:48 97:100]
            u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
            d(i,:) = v(i,:) + u(i,:) + u(i-1,:) + u(i-2,:); 
            opt_w_2N(:,i) = [1;1;1;1];
        end
        
        % Every N/2 switch to [-1 -1 -1 -1] as w0
        for i=[49:96]
            u(i,:) = v(i,:) - a1*u(i-1,:) - a2*u(i-2,:) - a3*u(i-3,:);
            d(i,:) = v(i,:) - u(i,:) - u(i-1,:) - u(i-2,:);  
            opt_w_2N(:,i) = [-1;-1;-1;-1];
        end

    lamda = 1;
    delta = 0.0001;

    % Storage variables

    % Starts at n=-1, followed by n=0
    curlyB = zeros(N+2,4); %time-by-orders
    % Starts at n=0
    curlyF = zeros(N+1,4); %time-by-orders
    % Starts at n=0
    gam = zeros(N+1,4);

    % All e's start at n=1
    ef = zeros(N,4);
    eb = zeros(N,4);
    em = zeros(N,4);

    % Starts at n=0
    pf = zeros(N+1,4);
    % Starts at n=0
    pb = zeros(N+1,4);
    % Starts at n=0, M goes up to 4
    pm = zeros(N,5);


    % Initialization Stage
    %   (a) Auxiliary parameter initialization. For order m=0,1,2 set
    % pf,m(0) = pb,m(0) = 0 (already done) and pm(0) = 0 [includes order 3]
    %   (b) Soft-constraint initialization. For order m=0,1,2,3 set
    % curlyBm(-1) = curlyBm(0) = delta
    curlyB(1:2,:) = delta^0.5;
    % curlyFm(0) = delta
    curlyF(1,:) = delta^0.5;
    %   (c) Data initialization. For n=1,2,... compute
    % ef,m=0(n) = eb,m=0(n) = u(n)
    ef(:,1) = u(1:N,:); % M=0 for all time
    eb(:,1) = u(1:N,:); % M=0 for all time
    % e,m=0(n) = d(n)
    em(:,1) = d(1:N,:); % M=0 for all time
    % gam,m=0(n) = 1
    gam(:,1) = 1^0.5;

    % Run QRD-LSL

    for m=2:4
        for n=2:N
            [curlyB(n-1+1,m-1),ef(n,m),pf(n,m-1),gam(n-1,m)] = Annil_b_m1_n1(lamda,curlyB(n-2+1,m-1),eb(n-1,m-1),ef(n,m-1),pf(n-1,m-1),gam(n-1,m-1));
            [curlyF(n,m-1),eb(n,m),pb(n,m-1)] = Annil_f_m1_n(lamda,curlyF(n-1,m-1),ef(n,m-1),eb(n-1,m-1),pb(n-1,m-1));
            [curlyB(n+1,m-1),em(n,m),pm(n,m-1)] = Annil_b_m1_n(lamda,curlyB(n-1+1,m-1),eb(n,m-1),em(n,m-1),pm(n-1,m-1));
        end
    end

    for n=2:N
        [curlyB(n-1+1,4),em(n,5),p(n,4)] = Annil_b_m1_n(lamda,curlyB(n-2+1,4),eb(n,4),em(n,4),pm(n-1,5));
    end

    % To find h regression coefficients: 
    % h_m-1(n) = p_m-1(n)./curlyB_m-1(n)
    h_1 = pm(:,1)./curlyB(3:end,1);
    h_2 = pm(:,2)./curlyB(3:end,2);
    h_3 = pm(:,3)./curlyB(3:end,3);
    h_4 = pm(:,4)./curlyB(3:end,4);

    w_hat = [h_1 h_2 h_3 h_4]';
    D_mc_2N(runs,:) = vecnorm(opt_w_2N - w_hat).^2;

end


figure
plot(mean(D_mc_Nov2))
hold on
plot(mean(D_mc_N))
hold on
plot(mean(D_mc_2N))
title('D Learning Curves for the QRD-LSL when nonstationarity. Oscillation between w0 = [1 1 1]^T and [-1 -1 -1]^T every N steps')
legend('Every N/2 = 12 steps osciallate','Every N = 24 steps osciallate','Every 2*N = 50 steps osciallate')
xlabel('Step N')

% There is a disturbance to the system but then the learning curve slopes
% downward. This shows how the system is very reactive and correcting
% toward outside shocks in the environment!

%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Below are all the functions needed for QRD-LSL

function [curlyB_m1_n1,ef_m_n,pf_m1_n,gam_m_n1] = Annil_b_m1_n1(lamda,curlyB_m1_n2,eb_m1_n1,ef_m1_n,pf_m1_n1,gam_m1_n1)
    % This function is for b,m-1(n-1)
    
    l = lamda^0.5;
    
    % Load prearray
    prearray = [l*curlyB_m1_n2 eb_m1_n1; l*pf_m1_n1 ef_m1_n; 0 gam_m1_n1];
    
    % Givens rotation
    c_b_m1_n1 = (l*curlyB_m1_n2);
    s_b_m1_n1 = eb_m1_n1;
    theta_b_m1_n1 = [c_b_m1_n1 -1*s_b_m1_n1; conj(s_b_m1_n1) c_b_m1_n1];
    
    % Postarray
    postarray = prearray*theta_b_m1_n1;
    
    % Unload postarray
    curlyB_m1_n1 = postarray(1,1);
    curlyB_m1_n1 = sqrt(curlyB_m1_n1);
    ef_m_n = postarray(2,2)./curlyB_m1_n1;
    pf_m1_n = postarray(2,1)./curlyB_m1_n1;
    gam_m_n1 = postarray(3,2)./curlyB_m1_n1;
end   

function [curlyF_m1_n,eb_m_n,pb_m1_n] = Annil_f_m1_n(lamda,curlyF_m1_n1,ef_m1_n,eb_m1_n1,pb_m1_n1)
    % This function is for f,m-1(n)
    
    l = lamda^0.5;
    
    % Load prearray
    prearray = [l*curlyF_m1_n1 ef_m1_n; l*pb_m1_n1 eb_m1_n1];
    
    % Givens rotation
    c_f_m1_n = (l*curlyF_m1_n1);
    s_f_m1_n = ef_m1_n;
    theta_f_m1_n = [c_f_m1_n -1*s_f_m1_n; conj(s_f_m1_n) c_f_m1_n];
    
    % Postarray
    postarray = prearray*theta_f_m1_n;
    
    % Unload postarray
    curlyF_m1_n = postarray(1,1);
    curlyF_m1_n = sqrt(curlyF_m1_n);
    eb_m_n = postarray(2,2)./curlyF_m1_n;
    pb_m1_n = postarray(2,1)./curlyF_m1_n;
    
end   


function [curlyB_m1_n,em_n,pm1_n] = Annil_b_m1_n(lamda,curlyB_m1_n1,eb_m1_n,em1_n,pm1_n1)
    % This function is for filtering
    
    l = lamda^0.5;
    
    % Load prearray
    prearray = [l*curlyB_m1_n1 eb_m1_n; l*pm1_n1 em1_n];
    
    % Givens rotation
    c_b_m1_n = (l*curlyB_m1_n1);
    s_b_m1_n = eb_m1_n;
    theta_b_m1_n = [c_b_m1_n -1*s_b_m1_n; conj(s_b_m1_n) c_b_m1_n];
    
    % Postarray
    postarray = prearray*theta_b_m1_n;
    
    % Unload postarray
    curlyB_m1_n = postarray(1,1);
    curlyB_m1_n = sqrt(curlyB_m1_n);
    em_n = postarray(2,2)./curlyB_m1_n;
    pm1_n = postarray(2,1)./curlyB_m1_n;
end 


%% (3) Function to implement core of QRD-RLS
% Computes w(n) directly from data matrix 
function [e_n,sqrt_phi_n,p_n_H] = QRD_RLS(u_n,d_n,lamda,sqrt_phi_nmin1,p_nmin1_H)
    l = lamda^0.5;
    [M,~] = size(u_n);
    
    % Setup prearray
    prearray = [l*sqrt_phi_nmin1 u_n; l*p_nmin1_H d_n; repmat(0,1,M) 1];
    
    % Annihilation using qr decomposition
    [Q,R] = qr(prearray'); %hermitian transpose A first because qr outputs an upper
    % triangular matrix R and we are looking for a lower triangular matrix.
    % By applying the hermitian transpose to R, we can obtain what we
    % require.
    
    % Postarray Extraction
    postarray = R';
 
    sqrt_phi_n = postarray(1:M,1:M); %upper left corner, M-by-M
    p_n_H = postarray(M+1,1:M); % 1-by-M
    ktzi_gamma = postarray(M+1,M+1);
    sqrt_gamma = postarray(end,end); %lower right corner
    
    % Compute e_n = ktzi*gamma
    e_n = ktzi_gamma*sqrt_gamma;

end   

%% (3) Function to implement core of Inverse QRD-RLS

function [w_n,sqrt_P_n,ktzi,sqrt_gamma] = inv_QRD_RLS(u_n,d_n,lamda,sqrt_P_nmin1,w_nmin1)
    l = lamda^-0.5;
    [M,~] = size(u_n);
    
    % Setup prearray
    prearray = [1 l*u_n'*sqrt_P_nmin1; repmat(0,M,1) l*sqrt_P_nmin1];
    
    % Annihilation using qr decomposition
    [Q,R] = qr(prearray'); %hermitian transpose A first because qr outputs an upper
    % triangular matrix R and we are looking for a lower triangular matrix.
    % By applying the hermitian transpose to R, we can obtain what we
    % require.
    
    % Postarray Extraction
    postarray = R';
 
    sqrt_gamma = postarray(1,1); %upper left corner
    k_gamma = postarray(M,1);
    sqrt_P_n = postarray(2:end,2:end); %bottom right corner
    
    % Compute ktzi and w in LMS update fashion
    k = k_gamma * inv(sqrt_gamma);
    ktzi = d_n - w_nmin1'*u_n;
    w_n = w_nmin1 + k*conj(ktzi);
end   

%% (3) Function to run QRD-RLS for arbitrary, unstructured 
% data vector u, does not need to be time series

function [e,sqrt_phi,p] = gen_QRD_RLS(u,d,M,lamda,delta)
    % Outputs: pH_n,sqrt_phi_n can be used to derive w_n
    
    % Convert u into matrix A of stacked u_M(n) column vectors
    A = toeplitz(u);
    A = A(1:M,M:end);

    [~,num_itr] = size(A);
    e = zeros(1,num_itr);
    
    % Initial conditions
    sqrt_phi_init = (delta^0.5).*eye(M);
    p_init = repmat(0,M,1)';
    
    sqrt_phi = sqrt_phi_init;
    p = p_init;
    
    for i=1:num_itr
        [e(1,i),sqrt_phi,p] = QRD_RLS(A(:,i),d(i+M-1,1),lamda,sqrt_phi,p);
    end
end

%% (3) Function to run inverse QRD-RLS for arbitrary, unstructured 
% data vector u, does not need to be time series

function [w,ktzi,sqrt_gamma] = gen_inv_QRD_RLS(u,d,M,lamda,delta,w_0)
    % Outputs: ktzi,sqrt_gamma can be used to derive e_n
    
    % Convert u into matrix A of stacked u_M(n) column vectors
    A = toeplitz(u);
    A = A(1:M,M:end);

    [~,num_itr] = size(A);
    
    w = zeros(M,num_itr);
    w(:,1) = w_0;
    
    % Initial conditions 
    P_init = (delta^-0.5).*eye(M);
    
    sqrt_P = P_init;
    
    for i=1:num_itr
        [w(:,i+1),sqrt_P,ktzi,sqrt_gamma] = inv_QRD_RLS(A(:,i),d(i+M-1,1),lamda,sqrt_P,w(:,i));
    end
end

%% (3) Function to run QRD-RLS for time series

function [e,sqrt_phi,p] = ts_QRD_RLS(u,d,M,lamda,delta,initialu)
    % initalu is an input argument that allows the user to input
    % u(2-M)...u(-2),u(-1),u(0) = M-1 values. Otherwise, prewindowing is
    % performed where u(n)=0 for n<1. If initialu is "inf", then
    % prewindowing is requested.
    
    if initialu == inf
        % Default prewindowing
        initialu = zeros(M-1,1);
    end
    
    % Convert u into matrix A of stacked u_M(n) column vectors
    A = toeplitz([initialu; u]);
    A = A(1:M,M:end);

    [~,num_itr] = size(A);
    e = zeros(1,num_itr);
    
    % Initial conditions
    sqrt_phi_init = (delta^0.5).*eye(M);
    p_init = repmat(0,M,1)';
    
    sqrt_phi = sqrt_phi_init;
    p = p_init;
    
    for i=1:num_itr
        [e(1,i),sqrt_phi,p] = QRD_RLS(A(:,i),d(i,1),lamda,sqrt_phi,p);
    end
    
    % Exact initialization requires e = 0 for n<=M
    e(1,1:M) = 0;
end

%% (3) Function to run time series inverse QRD-RLS 

function [w,ktzi,sqrt_gamma] = ts_inv_QRD_RLS(u,d,M,lamda,delta,initialu)
    % initalu is an input argument that allows the user to input
    % u(2-M)...u(-2),u(-1),u(0) = M-1 values. Otherwise, prewindowing is
    % performed where u(n)=0 for n<1. If initialu is "inf", then
    % prewindowing is requested.
    
    if initialu == inf
        % Default prewindowing
        initialu = zeros(M-1,1);
    end
    
    % Convert u into matrix A of stacked u_M(n) column vectors
    A = toeplitz([initialu; u]);
    A = A(1:M,M:end);

    [~,num_itr] = size(A);
    w = zeros(M,num_itr);
    
    % Initial conditions 
    P_init = (delta^-0.5).*eye(M);
    
    sqrt_P = P_init;
    
    for i=1:num_itr
        [w(:,i+1),sqrt_P,ktzi,sqrt_gamma] = inv_QRD_RLS(A(:,i),d(i,1),lamda,sqrt_P,w(:,i));
    end
end


%% (3) Function to compute w for QRD_RLS
function w = QRD_RLS_w(p_n_H,sqrt_phi)
    % Backsubstitution to convert sqrt_phi into phi^-0.5
    % Use the fact that sqrt_phi * inv_sqrt_phi = I to set up a reduced row
    % eschelon problem
    [dim,~] = size(sqrt_phi);
    backsub = [sqrt_phi eye(dim)];
    reduce = rref(backsub);
    inv_sqrt_phi = reduce(:,(0.5*end)+1:end);
    
    wH_n = p_n_H*inv_sqrt_phi;
    w = wH_n';
end

%% (3) Function to compute e for invQRD_RLS
function e = inv_QRD_RLS_e(ktzi,neg_sqrt_gamma)
    sqrt_gamma = (neg_sqrt_gamma)^-1; %scalar
    e = ktzi*sqrt_gamma*sqrt_gamma;
end


%% (2) Function to generate u and d signals 
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
    v1 = (1/sqrt(2))*(randn(length+6,1) + j*randn(length+6,1));
    v2 = (1/sqrt(2))*(randn(length+6,1) + j*randn(length+6,1));
    
    % Case 1
    if case_num == 1
        
        u = zeros(length+6,1);
        % Compute u[3]-u[6] because loop cant begin until n=7 due to constraints on
        % d[n] equation (explained below).
        u(3,:) = v1(3,:) - c1_a1*u(2,:) - c1_a2*u(1,:);
        u(4,:) = v1(4,:) - c1_a1*u(3,:) - c1_a2*u(2,:);
        u(5,:) = v1(5,:) - c1_a1*u(4,:) - c1_a2*u(3,:);
        u(6,:) = v1(6,:) - c1_a1*u(5,:) - c1_a2*u(4,:);

        d = zeros(length+6,1);
        % Initialize first 6 values of d1 = v2. To compute d[n], requires last 6
        % values of u because beta_0 is of length 6. Can't compute the first term
        % in d until n=7.
        d(1:6,:) = v2(1:6,:);

        % Compute u[n] = v1[n] - a1*u[n-1] - a2*u[n-2]
        for i=6:length+6
            u(i,:) = v1(i,:) - c1_a1*u(i-1,:) - c1_a2*u(i-2,:);
            d(i,:) = beta_0*flip(u(i-5:i,:)) + v2(i,:); %u1 must be flipped because 
            % u_m is defined as [u(n), u(n-1),... u(n-M+1)]'
        end 
        
        % Drop first 6 values
        u = u(7:end,:);
        d = d(7:end,:);
    end
    
    % Case 2
    if case_num == 2
        u = zeros(length+6,1);
        % Compute u[3]-u[6] because loop cant begin until n=7 due to constraints on
        % d[n] equation (explained below).
        u(3,:) = v1(3,:) - c2_a1*u(2,:) - c2_a2*u(1,:);
        u(4,:) = v1(4,:) - c2_a1*u(3,:) - c2_a2*u(2,:);
        u(5,:) = v1(5,:) - c2_a1*u(4,:) - c2_a2*u(3,:);
        u(6,:) = v1(6,:) - c2_a1*u(5,:) - c2_a2*u(4,:);

        d = zeros(length+6,1);
        % Initialize first 6 values of d2 = v2. To compute d[n], requires last 6
        % values of u because beta_0 is of length 6. Can't compute the first term
        % in d until n=7.
        d(1:6,:) = v2(1:6,:);

        % Compute u[n] = v1[n] - a1*u[n-1] - a2*u[n-2]
        for i=6:length+6
            u(i,:) = v1(i,:) - c2_a1*u(i-1,:) - c2_a2*u(i-2,:);
            d(i,:) = beta_0*flip(u(i-5:i,:)) + v2(i,:);
        end
        
        % Drop first 6 values
        u = u(7:end,:);
        d = d(7:end,:);
    end
end
