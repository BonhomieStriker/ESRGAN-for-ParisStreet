% loss_D = xlsread('loss_D.xlsx');
% loss_G = xlsread('loss_G.xlsx');
% PSNR = xlsread('PSNR.xlsx');
% Mean_PSNR = xlsread('Mean_PSNR.xlsx');
span = 3001:4000;
figure(1)
plot(span, loss_D(span), 'linewidth', 2)
hold on
plot(span, loss_G(span), 'linewidth', 2)
grid on 
legend('loss_D', 'loss_G')
xlabel('iter')
ylabel('loss')

figure(2)
plot(1:length(Mean_PSNR), Mean_PSNR, 'linewidth', 2)
grid on
xlabel('epoch')
ylabel('Mean PSNR')

figure(3)
plot(1:length(PSNR), PSNR)
grid on
xlabel('image')
ylabel('PSNR')
