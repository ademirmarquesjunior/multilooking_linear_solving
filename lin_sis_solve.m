function [x]= lin_sis_solve (im1, im2, im3, im4, im5, im6, im7, im8, im9);

syms p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25
eqn1 = p1 + p2 + p3 - p16 - p17 - p18  == (im1 - im4)*9;
eqn2 = p6 + p7 + p8 - p21 - p22 - p23  == (im4 - im7)*9;
eqn3 = p1 + p2 + p3 + p6 + p7 + p8 - p16 - p17 - p18 - p21 - p22 - p23 == (im1 - im7)*9;
eqn4 = p2 + p3 + p4 - p17 - p18 - p19  == (im2 - im5)*9;
eqn5 = p7 + p8 + p9 - p22 - p23 - p24  == (im5 - im8)*9;
eqn6 = p2 + p3 + p4 + p7 + p8 + p9 - p17 - p18 - p19 - p22 - p23 - p24 == (im2 - im8)*9;
eqn7 = p3 + p4 + p5 - p18 - p19 - p20  == (im3 - im6)*9;
eqn8 = p8 + p9 + p10 - p23 - p24 - p25  == (im6 - im9)*9;
eqn9 = p3 + p4 + p5 + p8 + p9 + p10 - p18 - p19 - p20 - p23 - p24 - p25 == (im3 - im9)*9;
eqn10 = p1 + p6 + p11 - p4 - p9 - p14  == (im1 - im2)*9;
eqn11 = p2 + p7 + p12 - p5 - p10 - p15  == (im2 - im3)*9;
eqn12 = p1 + p6 + p11 + p2 + p7 + p12 - p4 - p9 - p14 - p5 - p10 - p15 == (im1 - im3)*9;
eqn13 = p6 + p11 + p16 - p9 - p14 - p19  == (im4 - im5)*9;
eqn14 = p7 + p12 + p17 - p10 - p15 - p20  == (im5 - im6)*9;
eqn15 = p6 + p11 + p16 + p7 + p12 + p17 - p9 - p14 - p19 - p10 - p15 - p20  == (im4 - im6)*9;
eqn16 = p1 + p2 + p3 + p6 + p11 - p9 - p14 - p19 - p18 - p17  == (im1 - im5)*9;
eqn17 = p2 + p3 + p4 + p7 + p12 - p10 - p15  - p20 - p19 - p18  == (im2 - im6)*9;
eqn18 = p6 + p7 + p8 + p11 + p16 - p14 - p19  - p24 - p23 - p22  == (im4 - im8)*9;
eqn19 = p7 + p8 + p9 + p12 + p17 - p15 - p20  - p25 - p24 - p23  == (im5 - im9)*9;
eqn20 = p1 + p2 + p3 + p6 + p11 + p7 + p12 - p9 - p14 - p19 - p18 - p10 - p15 - p20  == (im1 - im6)*9;
eqn21 = p11 + p12 + p16 + p17 + p21 + p22 + p23 - p8 - p9 - p10 - p14 - p15 - p19 - p20  == (im7 - im6)*9;
eqn22 = p6 + p7 + p11 + p12 + p16 + p17 + p18 - p3 - p4 - p5 - p9 - p10 - p14 - p15  == (im4 - im3)*9;
eqn23 = p6 + p7 + p8 + p11 + p12 + p16 + p17 - p14 - p15 - p19 - p20 - p23 - p24 - p25  == (im4 - im9)*9;
eqn24 = p1 + p2 + p3 + p6 + p7 + p8 + p11 + p12 - p14 - p15 - p18 - p19 - p20 - p23 - p24 - p25  == (im1 - im9)*9;
eqn25 = p11 + p12 + p16 + p17 + p18 + p21 + p22 + p23 - p3 - p4 - p5 - p8 - p9 - p10 - p14 - p15  == (im7 - im3)*9;
eqn26 = p1 + p2 + p3 + p6 + p7 + p8 + p11 + p12 + p13 == im1*9;
eqn27 = p2 + p3 + p4 + p7 + p8 + p9 + p12 + p13 + p14 == im2*9;
eqn28 = p3 + p4 + p5 + p8 + p9 + p10 + p13 + p14 + p15 == im3*9;
eqn29 = p6 + p7 + p8 + p11 + p12 + p13 + p16 + p17 + p18 == im4*9;
eqn30 = p7 + p8 + p9 + p12 + p13 + p14 + p17 + p18 + p19 == im5*9;
eqn31 = p8 + p9 + p10 + p13 + p14 + p15 + p18 + p18 + p20 == im6*9;
eqn32 = p11 + p12 + p13 + p16 + p17 + p18 + p21 + p22 + p23 == im7*9;
eqn33 = p12 + p13 + p14 + p17 + p18 + p19 + p22 + p23 + p24 == im8*9;
eqn34 = p13 + p14 + p15 + p18 + p19 + p20 + p23 + p24 + p25 == im9*9;
% eqn35 = p1 == 83;
% eqn36 = p2 ==87;
% eqn37 = p3 ==81;
% eqn38 = p6 ==73;
% eqn39 = p7 ==64;
% eqn40 = p8 ==87;
% eqn41 = p11 ==72;
%, eqn35, eqn36, eqn37, eqn38, eqn39, eqn40, eqn41
[A,B] = equationsToMatrix([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8, eqn9, eqn10, eqn11, eqn12, eqn13, eqn14, eqn15, eqn16, eqn17, eqn18, eqn19, eqn20, eqn21, eqn22, eqn23, eqn24, eqn25, eqn26, eqn27, eqn28, eqn29, eqn30, eqn31, eqn32, eqn33, eqn34],...
[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25]);
%x = linsolve(A,B)
A=double(A);
B=double(B);
lb=zeros(25,1);
ub=255*ones(25,1);
x = lsqlin(A,B,[],[],[],[],lb,ub)

