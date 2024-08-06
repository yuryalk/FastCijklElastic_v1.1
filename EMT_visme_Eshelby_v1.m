tic;
clear,
figure(1),clf,colormap(jet), 
clc
format compact; format shortG
% Written by Yury Alkhimenkov
% Massachusetts Institute of Technology
% 28 Jan 2024

% This script is developed to run the CUDA C routine "FastCijklElastic_v1.cu" and visualize the results

% This script:
% 1) creates parameter files
% 2) compiles and runs the code on a GPU
% 3) visualize and save the result

clear 
%% set run_static=1 if you want to calculate the effective stiffnes tensor
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run_static = 1;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
format compact
% NBX is the input parameter to GPU
NBX = 6; 
NBY = 6; 
NBZ = 14;  

BLOCK_X  = 32; % BLOCK_X*BLOCK_Y<=1024
BLOCK_Y  = 4/2;
BLOCK_Z  = 8;
GRID_X   = NBX*1 *1;
GRID_Y   = NBY*4 *2*2;
GRID_Z   = NBZ*2 *2;

OVERLENGTH_X = 1;
OVERLENGTH_Y = OVERLENGTH_X;
OVERLENGTH_Z = OVERLENGTH_X;

nx = BLOCK_X*GRID_X  - OVERLENGTH_X; % size of the model in x
ny = BLOCK_Y*GRID_Y  - OVERLENGTH_Y; % size of the model in y
nz = BLOCK_Z*GRID_Z  - OVERLENGTH_Z; % size of the model in z

OVERX = OVERLENGTH_X;
OVERY = OVERLENGTH_Y;
OVERZ = OVERLENGTH_Z;

% dimensionally independent
Lx      = 1;                 % m
Ly      = 1*Lx;         % m
Lz      = 1.0*Lx;         % m
%% numerics
nt     = 1;
eiter  = 1e-12;
niter  = 1000*max(nx,ny)*0+nx*4;
Re     = 3*sqrt(10)/2*pi;
K_G    = 1; 

Cij_GPU = zeros(6,6);
%% preprocessing
dx        = Lx/(nx-1);
dy        = Ly/(ny-1);
dz        = Lz/(nz-1);
[x,y,z]     = ndgrid(-Lx/2:dx:Lx/2,-Ly/2:dy:Ly/2,-Lz/2:dz:Lz/2);
dt = 1;  dt = dt*1e-4;
CFL=(0.85 / sqrt(3));
Vpdt      = CFL*min(dx,dz);
nout   = 40;
layer_thkness =fix(nz/16/1.5/5.5  );
layer_thkness_ABS = (layer_thkness-1)*2+1;
rad0     =1.32   +0*0.02;
rad02    =1.36;
length_pores = 1/72 ;

if run_static==1
    DAT = 'double'; 
%% SOLID grains
G_1         =4.4*1e+10 / 1;% 3.6*1e10;
K_1         =3.6*1e+10 / 1;% 3.6*1e10;
eta_1       = 1e10  * 1e+10;
%% DRY pores and crack
%dt = 1;
G_2         =2* 1e6; % 
K_2         =1e-3* 1e9;
eta_2       = 1e20;

G_2_crack = G_2;
K_2_crack = K_2;
%G_2         = 1e-2*G_2;
%K_2         = 1e-2*K_2;
eta_3       =  eta_2;

G_2_solid   =G_2; K_2_solid  =K_2;eta_2_solid = eta_2;
%% SOLID SOLID; SOLID crack = HF
% G_2         =2* 1e6; K_2  =4.35* 1e9;eta_2   = 1e20;
% G_2_crack         =2   * 1e6; % 
% K_2_crack         =0.001*   1e9;
% eta_2_crack       = 1e20;
% 
% G_2         =2* 1e6; K_2  =4.3* 1e9;eta_2   = 1e20;
% G_2_crack         =2e-0*2   * 1e6; % 
% K_2_crack         = 1e-5*4.3*   1e9;
% eta_2_crack       = 1e20;

% G_3         = 2e-0*G_2;
% K_3         = 1e-5*K_2;
% eta_3       = 1e-5*eta_2;
%%%%G_2         =1e0* 1e6; K_2  =1e-3* 1e9;eta_2   = 1e20;

%%
% preproc
G           = G_1*ones(nx  ,ny,nz  ) ;  %G(rad<rad0) =  P0;
K           = K_1*ones(nx  ,ny,nz  ) ;  %G(rad<rad0) =  P0;
eta         = eta_1*ones(nx  ,ny,nz  ) ;
%%
length_cr = 1/4    *1;
aa = 1*length_cr;
bb = 1*length_cr;
cc = 2*length_cr;
length_cr = 1/4      *2;
aa = 1*length_cr;bb = 1*length_cr;cc = 2*length_cr;
rad   = sqrt((x-0.5).^2/aa^2+(y-0.5).^2 /bb^2 +1*(z-0*0.5).^2 /cc^2);rad0     =1.4;
%rad2   = sqrt((x-0.5).^2/aa^2+(y-0.5).^2 /bb^2 +0*(z-0*0.5).^2 /cc^2);%rad02     =1.46;
crack_radius = sqrt(0.5*rad02*rad02*aa^2);%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%G(rad<rad0) =  G_2; K(rad<rad0) =  K_2; eta(rad<rad0) =  eta_2;


Volume_incl = 4/3*pi*aa*bb*cc ;Volume_box = Lx*Ly*Lz;
PorosityIncl       = Volume_incl/Volume_box /1;
aspect_ratio = cc/aa;
%% crack moduli
% G(rad2<rad02) =  G_2_crack;K(rad2<rad02) =  K_2_crack;eta(rad2<rad02) =  eta_2;
% G(rad<rad0) =  G_2;  K(rad<rad0) =  K_2;  eta(rad<rad0) =  eta_2;
%     for i=1:fix(nz/2)-layer_thkness
%     G(:,:,i) =  G_1;    K(:,:,i) =  K_1;    eta(:,:,i) =  eta_1;
%     end
%     for i=fix(nz/2)+layer_thkness:nz
%     G(:,:,i) =  G_1;    K(:,:,i) =  K_1;    eta(:,:,i) =  eta_1;
%     end

G = G-G_2;
sumG = sum(G(:))/( (G_1-G_2)*nx*ny*nz); % volime of solid
porosity_crack = 1-sumG % porosity 0.0036135   0.0034177
G = G+G_2;

hh_grids = ((layer_thkness-1)*2 +1)
hh = hh_grids/nz; % 5 cells   rad0*rad0*aa^2
LL = sqrt(rad02*rad02*aa^2)*2 ;
crack_aspect_ratio = hh/LL
% spheres (pores) moduli

[x,y,z]     = ndgrid(-Lx/2:dx:Lx/2,-Ly/2:dy:Ly/2,-Lz/2:dz:Lz/2);
 aa = 10*length_pores *2;bb = 10*length_pores *2;cc = 1*length_pores *2;
 rad   = sqrt((x-0.5).^2/aa^2+(y-0.5).^2 /bb^2 +(z-0*0.5).^2 /cc^2);rad0     =1;
G(rad<rad0) =  G_2; K(rad<rad0) =  K_2; eta(rad<rad0) =  eta_2;

%[x,y,z]     = ndgrid(-Lx/2:dx:Lx/2,-Ly/2:dy:Ly/2,-Lz/2:dz:Lz/2);
% aa = 10*length_pores;bb = 10*length_pores;cc = 10*length_pores;
% rad   = sqrt((x-0.32).^2/aa^2+(y-0.32).^2 /bb^2 +(z-0*0.5).^2 /cc^2);rad0     =1;
% G(rad<rad0) =  G_2; K(rad<rad0) =  K_2; eta(rad<rad0) =  eta_2;
% rad   = sqrt((x+0.32).^2/aa^2+(y+0.32).^2 /bb^2 +(z-0*0.5).^2 /cc^2);rad0     =1;
% G(rad<rad0) =  G_2; K(rad<rad0) =  K_2; eta(rad<rad0) =  eta_2;
% rad   = sqrt((x+0.32).^2/aa^2+(y-0.32).^2 /bb^2 +(z-0*0.5).^2 /cc^2);rad0     =1;
% G(rad<rad0) =  G_2; K(rad<rad0) =  K_2; eta(rad<rad0) =  eta_2;
% rad   = sqrt((x-0.32).^2/aa^2+(y+0.32).^2 /bb^2 +(z-0*0.5).^2 /cc^2);rad0     =1;
% G(rad<rad0) =  G_2; K(rad<rad0) =  K_2; eta(rad<rad0) =  eta_2;

Volume_incl = 4/3*pi*aa*bb*cc ;
Volume_box = Lx*Ly*Lz;
PorosityIncl       = Volume_incl/Volume_box /4;
Porosity_pores       = 4*Volume_incl/Volume_box /1
pores_aspect_ratio = cc/aa
%%
dt = 1;
eta_vem  = 1./(1./eta + 1./((K_1+4/3*G_1).*dt)); 
fid      = fopen('eta_vem.dat','wb'); fwrite(fid, eta_vem(:),DAT); fclose(fid);
%%
%% 2D SLICE
figure(1);clf;colormap(jet);
subplot(211); imagesc(squeeze(G(:,:,fix(nz/2 )))'  );view(0,90);shading flat;axis square tight;colorbar;title('xy');pbaspect([1 1 1])
subplot(212); imagesc(squeeze(G(:,fix(ny/2 ),:))'  );view(0,90);shading flat;axis square tight;colorbar;title('xz');pbaspect([1 1 1])
drawnow
zzz = 1;
%% 3D plot
figure(2);clf;colormap(jet);
mincoh2 = min(G(:)); 
mincoh2=G_2;
set(gcf,'color','white');S1 = figure(2);clf; hold on;

isosurf = 1.00*mincoh2;%1.9*min(coh0(:));
is1  = isosurface(G, G_2);is2  = isosurface(G, G_2);
his1 = patch(is1); set(his1,'CData',+isosurf,'Facecolor','Flat','Edgecolor','none')
his2 = patch(is2); set(his2,'CData',+isosurf,'Facecolor','Flat','Edgecolor','none')

hold off; box on;xlabel('x, grid cells (-)'); ylabel('y, grid cells (-)');  zlabel('z, grid cells (-)');axis image; view(138,27)
camlight; camproj perspective
light('position',[0.6 -1 1]);     light('position',[0.8 -1.0 1.0]);

pos = get(gca,'position'); set(gca,'position',[0.01 pos(2), pos(3), pos(4)])
%h = colorbar; caxis([0.0101; 2*1e-2*G0]);
daspect([1,1,1]);  axis square; grid on;%cb = colorbar; set(cb,'position',[0.78, 0.16, 0.04, 0.165])
%title(cb,' cohesion c (-) ','Interpreter', 'latex');set(gca,'ColorScale','log')
grid on;%colormap(S1,Red_blue_colormap);
colormap(S1,jet(1000));ax = gca;ax.BoxStyle = 'full';%view(95,2);%view(61,16);
box on;ax.LineWidth = 0.5;xlim([0 nx]);ylim([0 ny]);zlim([0 nz]);title(['Oblate spheroid']);hold on;
%view(100,15);
view(-63,34);
% colormap([1 0 0]);
%%
%%
zzz = 1;
Re     = 3*sqrt(10)/2*pi;
K_G    =  K_1/G_1; 

fid      = fopen('etam.dat','wb'); fwrite(fid, eta(:),DAT); fclose(fid);
fid      = fopen('Km.dat','wb'); fwrite(fid, K(:),DAT); fclose(fid);
fid      = fopen('Gm.dat','wb'); fwrite(fid, G(:),DAT); fclose(fid);
%%
%% CUDA C33
TestN = 3;
Vx     = zeros(nx+1,ny,nz);Vy     = zeros(nx,ny+1,nz);Vz     = zeros(nx,ny,nz+1);
        Vx(1,:,:)       = 0*1e-4;         Vx(end,:,:)     = 0;
        Vy(:,1,:)       = 0*1e-4;         Vy(:,end,:)     = 0;
        Vz(:,:,1)       = 1*1e+4;         Vz(:,:,end)     = 0;
for jj=1:nx
            for ii=1:nx
                Vz(ii,jj,:)       = linspace(1e+4,0,nz+1);
            end
end
%DAT = 'single';
etaN=1e20; rhof0g=1;rhos0g=1; K_dry=1; B=1; alph=1; Mbiot=1; phi=1; rho_f=1; eb_new=1;iM22=1; eta_reg=1; K0=K_1;G0=G_1;
eta_ve  =1./(1./etaN + 1./((K_1+4/3*G_1).*dt)); nt = 1; 
pa1         = [ dt rhof0g rhos0g K_dry K0 B alph G0 etaN Mbiot phi rho_f eb_new iM22 nt eta_reg eta_ve TestN K_G];
fid         = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
fid      = fopen('Vx.dat','wb'); fwrite(fid, Vx(:),DAT); fclose(fid);
fid      = fopen('Vy.dat','wb'); fwrite(fid, Vy(:),DAT); fclose(fid);
fid      = fopen('Vz.dat','wb'); fwrite(fid, Vz(:),DAT); fclose(fid);
% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
code_name    = 'FastCijklElastic_v1';
run_cmd      =['nvcc -arch=sm_80 -O3' ...
    ' -DNBX='    int2str(NBX)        ' -DNBY='    int2str(NBY)       ' -DNBZ='    int2str(NBZ)         ...
    ' -DOVERX='  int2str(OVERX)      ' -DOVERY='  int2str(OVERY)     ' -DOVERZ='  int2str(OVERZ)       ...
    ' -DNPARS1=' int2str(length(0))  ' -DNPARS2=' int2str(length(0)) ' -DNPARS3=' int2str(length(0)) ' ',code_name,'.cu'];
system(run_cmd);
! a.exe
% Reading data
isave = nt-1;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end

name=[num2str(isave) '_0_residH.res']; id = fopen(name); residH  = fread(id,DAT); fclose(id); residH  = reshape(residH  ,3  ,1  ,1);
Av_Snorm = -residH(1) + residH(2);Av_Enorm = residH(3);
C33_GPU  = (Av_Snorm)./(Av_Enorm) ;

Cij_GPU(3,3) = C33_GPU;
%% CUDA C11
TestN = 1;
Vx     = zeros(nx+1,ny,nz);Vy     = zeros(nx,ny+1,nz);Vz     = zeros(nx,ny,nz+1);
        Vx(1,:,:)       = 1*1e+4;         Vx(end,:,:)     = 0;
        Vy(:,1,:)       = 0*1e-4;         Vy(:,end,:)     = 0;
        Vz(:,:,1)       = 0*1e-4;         Vz(:,:,end)     = 0;
for jj=1:nx
            for ii=1:nx
        Vx(:,ii,jj)       = linspace(1e+4,0,nx+1);
            end
end
pa1         = [ dt rhof0g rhos0g K_dry K0 B alph G0 etaN Mbiot phi rho_f eb_new iM22 nt eta_reg eta_ve TestN K_G];
fid         = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
fid      = fopen('Vx.dat','wb'); fwrite(fid, Vx(:),DAT); fclose(fid);
fid      = fopen('Vy.dat','wb'); fwrite(fid, Vy(:),DAT); fclose(fid);
fid      = fopen('Vz.dat','wb'); fwrite(fid, Vz(:),DAT); fclose(fid);
% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! a.exe
% Reading data
isave = nt-1;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end

name=[num2str(isave) '_0_residH.res']; id = fopen(name); residH  = fread(id,DAT); fclose(id); residH  = reshape(residH  ,3  ,1  ,1);
Av_Snorm = -residH(1) + residH(2);Av_Enorm = residH(3);
C11_GPU  = (Av_Snorm)./(Av_Enorm) ;

Cij_GPU(1,1) = C11_GPU;
%% CUDA C22
TestN = 2;
Vx     = zeros(nx+1,ny,nz);Vy     = zeros(nx,ny+1,nz);Vz     = zeros(nx,ny,nz+1);
        Vx(1,:,:)       = 0*1e-4;         Vx(end,:,:)     = 0;
        Vy(:,1,:)       = 1*1e+4;         Vy(:,end,:)     = 0;
        Vz(:,:,1)       = 0*1e-4;         Vz(:,:,end)     = 0;
for jj=1:nx
            for ii=1:nx
                Vy(ii,:,jj)       = linspace(1e+4,0,ny+1);
            end
end
eta_ve  =1./(1./etaN + 1./((K_1+4/3*G_1).*dt)); nt = 1;
pa1         = [ dt rhof0g rhos0g K_dry K0 B alph G0 etaN Mbiot phi rho_f eb_new iM22 nt eta_reg eta_ve TestN K_G];
fid         = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
fid      = fopen('Vx.dat','wb'); fwrite(fid, Vx(:),DAT); fclose(fid);
fid      = fopen('Vy.dat','wb'); fwrite(fid, Vy(:),DAT); fclose(fid);
fid      = fopen('Vz.dat','wb'); fwrite(fid, Vz(:),DAT); fclose(fid);
% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! a.exe
% Reading data
isave = nt-1;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end

name=[num2str(isave) '_0_residH.res']; id = fopen(name); residH  = fread(id,DAT); fclose(id); residH  = reshape(residH  ,3  ,1  ,1);
Av_Snorm = -residH(1) + residH(2);Av_Enorm = residH(3);
C22_GPU  = (Av_Snorm)./(Av_Enorm) ;

Cij_GPU(2,2) = C22_GPU;
%%
%% CUDA C12 C13 C23
TestN = 7;
Vx     = zeros(nx+1,ny,nz);Vy     = zeros(nx,ny+1,nz);Vz     = zeros(nx,ny,nz+1);
        Vx(1,:,:)       = 1*1e+4;         Vx(end,:,:)     = 0;
        Vy(:,1,:)       = 1*1e+4;         Vy(:,end,:)     = 0;
        Vz(:,:,1)       = 1*1e+4;         Vz(:,:,end)     = 0;
for jj=1:nx
            for ii=1:nx
                Vx(:,ii,jj)       = linspace(1e+4,0,nx+1);
                Vy(ii,:,jj)       = linspace(1e+4,0,ny+1);
                Vz(ii,jj,:)       = linspace(1e+4,0,nz+1);
            end
end
eta_ve  = 1./(1./etaN + 1./((K_1+4/3*G_1).*dt)); nt = 1;%K_G = 1;
pa1         = [ dt rhof0g rhos0g K_dry K0 B alph G0 etaN Mbiot phi rho_f eb_new iM22 nt eta_reg eta_ve TestN K_G];
fid         = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
fid      = fopen('Vx.dat','wb'); fwrite(fid, Vx(:),DAT); fclose(fid);
fid      = fopen('Vy.dat','wb'); fwrite(fid, Vy(:),DAT); fclose(fid);
fid      = fopen('Vz.dat','wb'); fwrite(fid, Vz(:),DAT); fclose(fid);
% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%system(run_cmd);
! a.exe
% Reading data
isave = nt-1;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end

name=[num2str(isave) '_0_residHxx.res']; id = fopen(name); residHxx  = fread(id,DAT); fclose(id); residHxx  = reshape(residHxx  ,3  ,1  ,1);
Av_Snorm_xx = -residHxx(1) + residHxx(2);Av_Enorm_xx = residHxx(3);
 
name=[num2str(isave) '_0_residHyy.res']; id = fopen(name); residHyy  = fread(id,DAT); fclose(id); residHyy  = reshape(residHyy  ,3  ,1  ,1);
Av_Snorm_yy = -residHyy(1) + residHyy(2);Av_Enorm_yy = residHyy(3);

name=[num2str(isave) '_0_residHzz.res']; id = fopen(name); residHzz  = fread(id,DAT); fclose(id); residHzz  = reshape(residHzz  ,3  ,1  ,1);
Av_Snorm_zz = -residHzz(1) + residHzz(2);Av_Enorm_zz = residHzz(3);
M_diag   = [Cij_GPU(1,1) 0 0; 0 Cij_GPU(2,2) 0; 0 0 Cij_GPU(3,3)];
M_strain = [Av_Enorm_xx; Av_Enorm_yy; Av_Enorm_zz ];
M_stress = [Av_Snorm_xx; Av_Snorm_yy; Av_Snorm_zz ];

M_Sdev = (eye(3).*M_stress - M_diag.*M_strain);M_E    = (eye(3).*M_strain);
% %
% %%%
% %syms C12 C13 C23
% 
% M_diag   = [Cij_GPU(1,1) 0 0; 0 Cij_GPU(2,2) 0; 0 0 Cij_GPU(3,3)];
% M_strain = [Av_Enorm_xx; Av_Enorm_yy; Av_Enorm_zz ];
% M_stress = [Av_Snorm_xx; Av_Snorm_yy; Av_Snorm_zz ];
% 
% % M_Cij   = [0 C12 C13; C12 0 C23; C13 C23 0];
% % 
% % M_stress - M_diag*M_strain =  M_Cij*M_strain;
% M_Sdev = (eye(3).*M_stress - M_diag.*M_strain);
% M_E    = (eye(3).*M_strain);
% 
% syms C12 C13 C23 M_E22 M_E11 M_E33 M_Sdev11 M_Sdev22 M_Sdev33
% eqn1 = M_E22*C12 + M_E33*C13 == M_Sdev11;
% eqn2 = M_E11*C12 + M_E33*C23 == M_Sdev22;
% eqn3 = M_E11*C13 + M_E22*C23 == M_Sdev33;
% 
% sol = solve([eqn1, eqn2, eqn3], [C12, C13, C23]);
% C12s = sol.C12
% C13s = sol.C13
% C23s = sol.C23
% % 

%M_Cij = (M_stress - M_diag.*M_strain)\M_strain*2;
%M_Cij = (eye(3).*M_strain)\M_dev;
C12 = (M_E(1,1)*M_Sdev(1,1) + M_E(2,2)*M_Sdev(2,2) - M_E(3,3)*M_Sdev(3,3))/(2*M_E(1,1)*M_E(2,2));
C13 =(M_E(1,1)*M_Sdev(1,1) - M_E(2,2)*M_Sdev(2,2) + M_E(3,3)*M_Sdev(3,3))/(2*M_E(1,1)*M_E(3,3));
C23 =(M_E(2,2)*M_Sdev(2,2) - M_E(1,1)*M_Sdev(1,1) + M_E(3,3)*M_Sdev(3,3))/(2*M_E(2,2)*M_E(3,3));

Cij_GPU(1,2) =  C12;  Cij_GPU(1,3) =  C13;  Cij_GPU(2,3) =  C23;
     for i=1:6
        for j=i:6
      Cij_GPU(j,i) = Cij_GPU(i,j);
        end
     end
%%
%% CUDA C66 XY
dt     = 1;
TestN = 4;
eta_vem1  = 1./(1./eta_1  + 1./(G.*dt));
eta_vem  = 1./(1./eta_1 + 1./((K +4/3*G ).*dt)); 
eta_vem = precond2(eta_vem);
%eta_vem = smooth3(eta_vem,"box",3);
eta_vem  = gather(eta_vem);
fid      = fopen('eta_vem.dat','wb'); fwrite(fid, eta_vem(:),DAT); fclose(fid);
Gsaved = G ; Ksaved = K ;
Vx     = zeros(nx+1,ny,nz);Vy     = zeros(nx,ny+1,nz);Vz     = zeros(nx,ny,nz+1);
for jj=1:nx
    for ii=1:nx+1
        Vy(:,ii,jj)       = linspace(1e+4,0,nx);
    end
end
Vy(1,:,:)       = 1e+4;         Vy(end,:,:)     = 0;
eta_ve  = 1./(1./etaN + 1./((K_1+4/3*G_1).*dt)); nt = 1;
pa1         = [ dt rhof0g rhos0g K_dry K0 B alph G0 etaN Mbiot phi rho_f eb_new iM22 nt eta_reg eta_ve TestN K_G];
fid         = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
%fid      = fopen('Gm.dat','wb'); fwrite(fid, G(:),DAT); fclose(fid);
%fid      = fopen('Km.dat','wb'); fwrite(fid, K(:),DAT); fclose(fid);
fid      = fopen('Vx.dat','wb'); fwrite(fid, Vx(:),DAT); fclose(fid);
fid      = fopen('Vy.dat','wb'); fwrite(fid, Vy(:),DAT); fclose(fid);
fid      = fopen('Vz.dat','wb'); fwrite(fid, Vz(:),DAT); fclose(fid);
% Running on GPU
! a.exe
% Reading data
isave = nt-1;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end

name=[num2str(isave) '_0_residH.res']; id = fopen(name); residH  = fread(id,DAT); fclose(id); residH  = reshape(residH  ,3  ,1  ,1);
Av_Sxy =  residH(1) ;Av_Exy = residH(3);
C66_GPU  = (Av_Sxy)./(Av_Exy)/2;

Cij_GPU(6,6) = C66_GPU;
%% CUDA C55 XZ
TestN = 5;
Vx     = zeros(nx+1,ny,nz);Vy     = zeros(nx,ny+1,nz);Vz     = zeros(nx,ny,nz+1);
for jj=1:nx+1
    for ii=1:nx
        Vz(:,ii,jj)       = linspace(1e+4,0,nx);
    end
end
Vz(1,:,:)       = 1e+4;         Vz(end,:,:)     = 0;
eta_ve  = 1./(1./etaN + 1./((K_1+4/3*G_1).*dt)); nt = 1;
pa1         = [ dt rhof0g rhos0g K_dry K0 B alph G0 etaN Mbiot phi rho_f eb_new iM22 nt eta_reg eta_ve TestN K_G];
fid         = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
%fid      = fopen('Gm.dat','wb'); fwrite(fid, G(:),DAT); fclose(fid);
%fid      = fopen('Km.dat','wb'); fwrite(fid, K(:),DAT); fclose(fid);
fid      = fopen('Vx.dat','wb'); fwrite(fid, Vx(:),DAT); fclose(fid);
fid      = fopen('Vy.dat','wb'); fwrite(fid, Vy(:),DAT); fclose(fid);
fid      = fopen('Vz.dat','wb'); fwrite(fid, Vz(:),DAT); fclose(fid);
% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! a.exe
% Reading data
isave = nt-1;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end

name=[num2str(isave) '_0_residH.res']; id = fopen(name); residH  = fread(id,DAT); fclose(id); residH  = reshape(residH  ,3  ,1  ,1);
Av_Sxz =  residH(1) ;Av_Exz = residH(3);
C55_GPU  = (Av_Sxz)./(Av_Exz)/2;

Cij_GPU(5,5) = C55_GPU;
%% CUDA C44 YZ
TestN = 6;
G = Gsaved ;
K = Ksaved ;
Vx     = zeros(nx+1,ny,nz);Vy     = zeros(nx,ny+1,nz);Vz     = zeros(nx,ny,nz+1);
for jj=1:nx+1
    for ii=1:nx
        Vz(ii,:,jj)       = linspace(1e+4,0,nx);
    end
end
Vz(:,1,:)       = 1e+4;         Vz(:,end,:)     = 0;
eta_ve  = 1./(1./etaN + 1./((K_1+4/3*G_1).*dt)); nt = 1;
pa1         = [ dt rhof0g rhos0g K_dry K0 B alph G0 etaN Mbiot phi rho_f eb_new iM22 nt eta_reg eta_ve TestN K_G];
fid         = fopen('pa1.dat','wb'); fwrite(fid,pa1(:),DAT); fclose(fid);
fid      = fopen('Gm.dat','wb'); fwrite(fid, G(:),DAT); fclose(fid);
fid      = fopen('Km.dat','wb'); fwrite(fid, K(:),DAT); fclose(fid);
fid      = fopen('Vx.dat','wb'); fwrite(fid, Vx(:),DAT); fclose(fid);
fid      = fopen('Vy.dat','wb'); fwrite(fid, Vy(:),DAT); fclose(fid);
fid      = fopen('Vz.dat','wb'); fwrite(fid, Vz(:),DAT); fclose(fid);
% Running on GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! a.exe
% Reading data
isave = nt-1;
infos = load('0_infos.inf');  PRECIS=infos(1); nx=infos(2); ny=infos(3); nz=infos(4); NB_PARAMS=infos(5); NB_EVOL=infos(6);
if (PRECIS==8), DAT = 'double';  elseif (PRECIS==4), DAT = 'single';  end

name=[num2str(isave) '_0_residH.res']; id = fopen(name); residH  = fread(id,DAT); fclose(id); residH  = reshape(residH  ,3  ,1  ,1);
Av_Syz =  residH(1) ;Av_Eyz = residH(3);
C44_GPU  = (Av_Syz)./(Av_Eyz)/2;

Cij_GPU(4,4) = C44_GPU;
%%

%%
%% END GPU
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%format shortG
Kmatrix= K_1;
Gmatrix= G_1;
Kincl  = K_2;
Gincl  = G_2;
alfa   = pores_aspect_ratio;
phi    = PorosityIncl;

C_eff_Nonit =[
   9.4425e+10   6.6482e+09   6.5653e+09            0            0            0
   6.6482e+09   9.4425e+10   6.5653e+09            0            0            0
   6.5653e+09   6.5653e+09   9.1929e+10            0            0            0
            0            0            0   4.3283e+10            0            0
            0            0            0            0   4.3283e+10            0
            0            0            0            0            0   4.3888e+10];

C_eff_GPU   =  (Cij_GPU)
Kbulk_Nonit = 1/9*( C_eff_Nonit(1,1) + C_eff_Nonit(2,2)+C_eff_Nonit(3,3)+2*(C_eff_Nonit(1,2) + C_eff_Nonit(1,3)+C_eff_Nonit(2,3)) )
Kbulk_GPU   = 1/9*( C_eff_GPU(1,1) + C_eff_GPU(2,2)+C_eff_GPU(3,3)+2*(C_eff_GPU(1,2) + C_eff_GPU(1,3)+C_eff_GPU(2,3)) )
Kbulk_GPU_direct = 1/3*(Av_Snorm_xx+ Av_Snorm_yy+ Av_Snorm_zz)./(Av_Enorm_xx+ Av_Enorm_yy+ Av_Enorm_zz);

ABS_diff_Num_An =  (C_eff_Nonit - Cij_GPU)
Persent_diff_Num_An =  ( (C_eff_Nonit - Cij_GPU)./C_eff_Nonit*100 );
Persent_diff_Num_An(isnan(Persent_diff_Num_An))=0;

Error_Percent_averadge = sum(Persent_diff_Num_An(:))/36

%%
figure(4); clf;
ii = 1:9; 
GPU_data = [ C_eff_GPU(1,1) C_eff_GPU(2,2) C_eff_GPU(3,3) C_eff_GPU(4,4) C_eff_GPU(5,5) C_eff_GPU(6,6)...
    C_eff_GPU(1,2) C_eff_GPU(1,3) C_eff_GPU(2,3) ] ;
maxGPU = max(GPU_data(:));
GPU_data = GPU_data./max(GPU_data(:));
Nonit_data = [ C_eff_Nonit(1,1) C_eff_Nonit(2,2) C_eff_Nonit(3,3) C_eff_Nonit(4,4) C_eff_Nonit(5,5) C_eff_Nonit(6,6)...
    C_eff_Nonit(1,2) C_eff_Nonit(1,3) C_eff_Nonit(2,3) ]./ maxGPU;

col = [0  0  0 ];
plot(ii, GPU_data,'o','color',col,...
    'MarkerEdgeColor',col,...
    'MarkerFaceColor',col,'MarkerSize',10,...
    'LineWidth',2); grid on;hold on; 
col = [0.99 0.3 0.3];
plot(ii, Nonit_data,'x','color',col,...
    'MarkerEdgeColor',col,...
    'MarkerFaceColor',col,'MarkerSize',10,...
    'LineWidth',2);  

x_range = [1 2 3 4 5 6 7 8 9];
set(gca,'xtick',x_range); 
xticklabels({'C_{11}','C_{22}','C_{33}', 'C_{44}', 'C_{55}', 'C_{66}', 'C_{12}', 'C_{13}', 'C_{23}'})

xlabel('Components of the stiffness matrix C_{mn}'),ylabel('C_{mn}/ max(C_{mn})'); xlim([0 10]); ylim([0 1.4]);
legend('Numerical solution: FastCijklElastic{\_}GPU3D','Exact analytical solution');
title(['Oblate spheroid: cumulative error = 0.096 %']);

if 4>3; return; end
else
%%
end

%%
zzz = 1;
%delete  a.exe *.dat *.res *.inf
%%
%%
function Ap = precond2(A)
Ap     = A;
Ap     = max(max(max(max(max(  Ap(1:end-1,1:end-1,1:end-1) ,Ap(1:end-1,1:end-1,2:end) ),Ap(1:end-1,2:end,1:end-1)),Ap(1:end-1,2:end,2:end)) ...
        ,            Ap(2:end  ,1:end-1,2:end)),Ap(2:end  ,2:end,2:end));
Ap     = max(max(max(Ap(1:end-1,1:end-1,1:end-1) ,Ap(1:end-1,2:end,1:end-1)) ...
    ,                Ap(2:end  ,1:end-1,2:end)),Ap(2:end  ,2:end,2:end));

%Ap     = max(max(max(Ap(1:end-1,1:end-1,1:end-1) ,Ap(1:end-1,2:end,1:end-1)) ...
%    ,                Ap(2:end  ,1:end-1,2:end)),Ap(2:end  ,2:end,2:end));
%Ap     = (expx(expy(expz(Ap))));
A(2:end-1,2:end-1,2:end-1) = Ap;
A([1 end],2:end-1,2:end-1) = Ap([1 end],:,:);
A(2:end-1,[1 end],2:end-1) = Ap(:,[1 end],:);
A(2:end-1,2:end-1,[1 end]) = Ap(:,:,[1 end]);
MinA = min(A(:));
MaxA = 1.0*max(A(:));

diff_stepen = abs(log10(MinA)-(log10(MaxA)))*0.76;%68
%Sign_min=log10(MinA)/abs(log10(MinA));
%Sign_max=log10(MaxA)/abs(log10(MaxA));
cut = (10^diff_stepen)*MinA;
Ap = A;%  + (10^diff_stepen)*MinA;
Ap(Ap<cut)=Ap(Ap<cut)+cut;
P = Ap;
npow  = 4;
Pmax  = max(P(:));
P     = P/Pmax;
laplP = zeros(size(P));
laplP(2:end-1,2:end-1,2:end-1) = diff(P(:,2:end-1,2:end-1).^npow/npow,2,1) ...
    +                    diff(P(2:end-1,:,2:end-1).^npow/npow,2,2) +  diff(P(2:end-1,2:end-1,:).^npow/npow,2,3);
P                      = P + 1/16*laplP;
P(:,[1 end],:)           = 1*P(:,[2 (end-1)],:);
P([1 end],:,:)           = 1*P([2 (end-1)],:,:);
P(:,:,[1 end])           = 1*P(:,:,[2 (end-1)]);
%P     = P*Pmax;
%Ap     = P/max(P(:))*Pmax;
P     = P*Pmax;
%Ap     = P/max(P(:))*MaxA;
Ap     = P/max(P(:))*Pmax;

end


function Ap = precond3(A) %STATIC tests, eta = 1e20
Ap     = A;
Ap     = max(max(max(max(max(  Ap(1:end-1,1:end-1,1:end-1) ,Ap(1:end-1,1:end-1,2:end) ),Ap(1:end-1,2:end,1:end-1)),Ap(1:end-1,2:end,2:end)) ...
        ,            Ap(2:end  ,1:end-1,2:end)),Ap(2:end  ,2:end,2:end));
Ap     = max(max(max(Ap(1:end-1,1:end-1,1:end-1) ,Ap(1:end-1,2:end,1:end-1)) ...
    ,                Ap(2:end  ,1:end-1,2:end)),Ap(2:end  ,2:end,2:end));

%Ap     = max(max(max(Ap(1:end-1,1:end-1,1:end-1) ,Ap(1:end-1,2:end,1:end-1)) ...
%    ,                Ap(2:end  ,1:end-1,2:end)),Ap(2:end  ,2:end,2:end));
%Ap     = (expx(expy(expz(Ap))));
A(2:end-1,2:end-1,2:end-1) = Ap;
A([1 end],2:end-1,2:end-1) = Ap([1 end],:,:);
A(2:end-1,[1 end],2:end-1) = Ap(:,[1 end],:);
A(2:end-1,2:end-1,[1 end]) = Ap(:,:,[1 end]);
MinA = min(A(:));
MaxA = 1.0*max(A(:));

diff_stepen = abs(log10(MinA)-(log10(MaxA)))*0.88;
%Sign_min=log10(MinA)/abs(log10(MinA));
%Sign_max=log10(MaxA)/abs(log10(MaxA));

Ap = A;%  + (10^diff_stepen)*MinA;

P = Ap;
npow  = 4;
Pmax  = max(P(:));
P     = P/Pmax;
laplP = zeros(size(P));
laplP(2:end-1,2:end-1,2:end-1) = diff(P(:,2:end-1,2:end-1).^npow/npow,2,1) ...
    +                    diff(P(2:end-1,:,2:end-1).^npow/npow,2,2) +  diff(P(2:end-1,2:end-1,:).^npow/npow,2,3);
P                      = P + 1/16*laplP;
P(:,[1 end],:)           = 1*P(:,[2 (end-1)],:);
P([1 end],:,:)           = 1*P([2 (end-1)],:,:);
P(:,:,[1 end])           = 1*P(:,:,[2 (end-1)]);
%P     = P*Pmax;
%Ap     = P/max(P(:))*Pmax;
P     = P*Pmax;
%Ap     = P/max(P(:))*MaxA;
Ap     = P/max(P(:))*Pmax;

end
function Ap=gpuArray(A)
   Ap= (A);
end