import DataFrames, LinearAlgebra
Vb = 4.16/sqrt(3); Sb = 1000; Ib = Sb/Vb; Zb = ((1000*Vb)^2)/(Sb*1000);
branch_data = DataFrames.DataFrame([
    (1,  2,  1, 1850), (2,  3,  1, 960), (3,  24, 1, 400),
    (3,  27, 3, 360), (3,  4,  2, 1320), (4,  5,  4, 240),
    (4,  9,  3, 600), (5,  6,  3, 280), (6,  7,  4, 200),
    (6,  8,  4, 280), (9,  10, 3, 200), (10, 23, 3, 600),
    (10, 11, 3, 320), (11, 13, 3, 320), (11, 12, 4, 320),
    (13, 14, 3, 560), (14, 18, 3, 640), (14, 15, 4, 520),
    (15, 16, 4, 200), (15, 17, 4, 1280), (18, 19, 3, 400),
    (19, 20, 3, 400), (20, 22, 3, 400), (20, 21, 4, 200),
    (24, 26, 4, 320), (24, 25, 4, 240), (27, 28, 3, 520),
    (28, 29, 4, 80), (28, 31, 3, 800), (29, 30, 4, 520),
    (31, 34, 4, 920), (31, 32, 3, 600), (32, 33, 4, 280),
    (34, 36, 4, 760), (34, 35, 4, 120)
]); 
DataFrames.rename!(branch_data, [:i, :j, :Zij, :Lij]);
branch_data.Lij = branch_data.Lij/3280.8399; 
node_data = DataFrames.DataFrame([
    (1,  0,   0,  0,   0,  0,   0,   0),
    (2,  140, 70, 140, 70, 350, 175, 0),
    (3,  0,   0,  0,   0,  0,   0,   0),
    (4,  0,   0,  0,   0,  0,   0,   0),
    (5,  0,   0,  0,   0,  42,  21,  0),
    (6,  42,  21, 0,   0,  0,   0,   0),
    (7,  42,  21, 42,  21, 42,  21,  0),
    (8,  42,  21, 0,   0,  0,   0,   0),
    (9,  0,   0,  0,   0,  85,  40,  0),
    (10, 0,   0,  0,   0,  0,   0,   0),
    (11, 0,   0,  0,   0,  0,   0,   0),
    (12, 0,   0,  0,   0,  42,  21,  0),
    (13, 85,  40, 0,   0,  0,   0,   0),
    (14, 0,   0,  0,   0,  42,  21,  0),
    (15, 0,   0,  0,   0,  0,   0,   0),
    (16, 0,   0,  0,   0,  85,  40,  0),
    (17, 0,   0,  42,  21, 0,   0,   0),
    (18, 140, 70, 0,   0,  0,   0,   0),
    (19, 126, 62, 0,   0,  0,   0,   0),
    (20, 0,   0,  0,   0,  0,   0,   0),
    (21, 0,   0,  0,   0,  85,  40,  0),
    (22, 0,   0,  0,   0,  42,  21,  0),
    (23, 0,   0,  85,  40, 0,   0,   0),
    (24, 0,   0,  0,   0,  0,   0,   0),
    (25, 0,   0,  0,   0,  85,  40,  0),
    (26, 8,   4,  85,  40, 0,   0,   0),
    (27, 0,   0,  0,   0,  85,  40,  0),
    (28, 0,   0,  0,   0,  0,   0,   0),
    (29, 17,  8,  21,  10, 0,   0,   0),
    (30, 85,  40, 0,   0,  0,   0,   0),
    (31, 0,   0,  0,   0,  85,  40,  0),
    (32, 0,   0,  0,   0,  0,   0,   0),
    (33, 0,   0,  42,  21, 0,   0,   0),
    (34, 0,   0,  0,   0,  0,   0,   0),
    (35, 0,   0,  140, 70, 21,  10,  0),
    (36, 0,   0,  42,  21, 0,   0,   0),
]);
DataFrames.rename!(node_data, [:i,:Pai, :Qai, :Pbi, :Qbi, :Pci, :Qci, :Type])
daily_data = DataFrames.DataFrame([
    (1,0.318840579710145), (2,0.231884057971015),
    (3,0.217391304347826), (4,0.173913043478261),
    (5,0.188405797101449), (6,0.246376811594203),
    (7,0.318840579710145), (8,0.463768115942029),
    (9,0.666666666666667), (10,0.782608695652174),
    (11,0.884057971014493), (12,0.942028985507247),
    (13,0.985507246376812), (14,0.898550724637681),
    (15,0.913043478260870), (16,0.927536231884058),
    (17,0.927536231884058), (18,0.927536231884058),
    (19,0.884057971014493), (20,1),
    (21,1),                 (22,0.898550724637681),
    (23,0.739130434782609), (24,0.565217391304348)
]);
DataFrames.rename!(daily_data, [:t, :CD]);
NN = size(node_data,1); NL = size(branch_data,1); A3 = zeros(3*NN,3*NL);
Sdy = complex(zeros(3*(NN),1)); Sdd = complex(zeros(3*(NN),1));
for k = 1:NN
    if node_data.Type[k] == 0
        Sdy[3*k-2:3*k,1] = [node_data.Pai[k] + im*node_data.Qai[k];
                            node_data.Pbi[k] + im*node_data.Qbi[k];
                            node_data.Pci[k] + im*node_data.Qci[k]]/Sb; 
    else    
        Sdd[3*k-2:3*k,1] = [node_data.Pai[k] + im*node_data.Qai[k];
                            node_data.Pbi[k] + im*node_data.Qbi[k];
                            node_data.Pci[k] + im*node_data.Qci[k]]/Sb; 
    end
end
for l = 1:NL
    Ni = branch_data.i[l]; Nj = branch_data.j[l];
    A3[3*Ni-2:3*Ni,3*l-2:3*l] = [1 0 0; 0 1 0; 0 0 1];
    A3[3*Nj-2:3*Nj,3*l-2:3*l] = [-1 0 0; 0 -1 0; 0 0 -1];
end
conductor_data = DataFrames.DataFrame([
    (1, 180, 1986), (2, 200, 2790),
    (3, 230, 3815), (4, 270, 5090),
    (5, 300, 8067), (6, 340, 12673),
    (7, 600, 23419),(8, 720, 30070),
]);
DataFrames.rename!(conductor_data, [:caliber, :Imax, :Cinv]);
conductor_data.Imax = conductor_data.Imax/Ib;
Z1 = [1.1093 + im*1.0111 0.0592 + im*0.4876 0.0592 + im*0.4618;
      0.0592 + im*0.4876 1.1093 + im*1.0111 0.0592 + im*0.5550;
      0.0592 + im*0.4618 0.0592 + im*0.5550 1.1093 + im*1.0111];
Z2 = [0.9167 + im*1.0111 0.0592 + im*0.4876 0.0592 + im*0.4618;
      0.0592 + im*0.4876 0.9167 + im*1.0111 0.0592 + im*0.5550;
      0.0592 + im*0.4618 0.0592 + im*0.5550 0.9167 + im*1.0111];
Z3 = [0.7551 + im*1.0063 0.0592 + im*0.4876 0.0592 + im*0.4618
      0.0592 + im*0.4876 0.7551 + im*1.0063 0.0592 + im*0.5550
      0.0592 + im*0.4618 0.0592 + im*0.5550 0.7551 + im*1.0063];
Z4 = [0.6153 + im*0.9961 0.0592 + im*0.4876 0.0592 + im*0.4618;
      0.0592 + im*0.4876 0.6153 + im*0.9961 0.0592 + im*0.5550;
      0.0592 + im*0.4618 0.0592 + im*0.5550 0.6153 + im*0.9961];
Z5 = [0.5084 + im*0.9839 0.0592 + im*0.4876 0.0592 + im*0.4618;
      0.0592 + im*0.4876 0.5084 + im*0.9839 0.0592 + im*0.5550;
      0.0592 + im*0.4618 0.0592 + im*0.5550 0.5084 + im*0.9839];  
Z6 = [0.4270 + im*0.9609 0.0592 + im*0.4876 0.0592 + im*0.4618;
      0.0592 + im*0.4876 0.4270 + im*0.9609 0.0592 + im*0.5550;
      0.0592 + im*0.4618 0.0592 + im*0.5550 0.4270 + im*0.9609];
Z7 = [0.2201 + im*0.8683 0.0592 + im*0.4876 0.0592 + im*0.4618;
      0.0592 + im*0.4876 0.2201 + im*0.8683 0.0592 + im*0.5550;
      0.0592 + im*0.4618 0.0592 + im*0.5550 0.2201 + im*0.8683];
Z8 = [0.1747 + im*0.8593 0.0592 + im*0.4876 0.0592 + im*0.4618;
      0.0592 + im*0.4876 0.1747 + im*0.8593 0.0592 + im*0.5550;
      0.0592 + im*0.4618 0.0592 + im*0.5550 0.1747 + im*0.8593];
Zp3 = [Z1;Z2;Z3;Z4;Z5;Z6;Z7;Z8];
Zp3 = Zp3/Zb; NC = size(conductor_data, 1);
using JuMP, AmplNLWriter, Bonmin_jll
TPCSMP = Model(() -> AmplNLWriter.Optimizer(Bonmin_jll.amplexe))
set_attribute(TPCSMP, "bonmin.nlp_log_level", 0)
set_attribute(TPCSMP, "honor_original_bounds", "yes")
M = [1 -1 0; 0 1 -1; -1 0 1]; slack = 1;
Vmin = 0.90; Vmax = 1.10;
Imin = 0.0; Imax = conductor_data.Imax;
H = size(daily_data,1); dh = 24/H;
CkWh = 0.1390; T = 365;
@variable(TPCSMP,V[k in 1:3*NN, h in 1:H] in ComplexPlane());
for h = 1:H 
    for k = 1:NN 
        set_start_value(real(V[3*k-2,h]),1.0);
        set_start_value(real(V[3*k-1,h]),-0.5);
        set_start_value(real(V[3*k,h]),-0.5);
        set_start_value(imag(V[3*k-2,h]),0.0);
        set_start_value(imag(V[3*k-1,h]),-0.866025403784439);
        set_start_value(imag(V[3*k,h]),0.866025403784439);
    end
end
@variable(TPCSMP, Sg[k in 1:3*NN, h in 1:H] in ComplexPlane());
@variable(TPCSMP, Ig[k in 1:3*NN, h in 1:H] in ComplexPlane());
@variable(TPCSMP, Idy[k in 1:3*NN, h in 1:H] in ComplexPlane());
@variable(TPCSMP, Idd[k in 1:3*NN, h in 1:H] in ComplexPlane());
@variable(TPCSMP, J[l in 1:3*NL, h in 1:H] in ComplexPlane());
@variable(TPCSMP, Vj[l in 1:3*NL, h in 1:H] in ComplexPlane());
@variable(TPCSMP, Ilmax[l in 1:NL]);
@variable(TPCSMP, Y[l in 1:NL, p in 1:NC], Bin);
@variable(TPCSMP, Zloss); @variable(TPCSMP, Zinv);
for l = 1:NL        
    @constraint(TPCSMP, Ilmax[l] == sum(Y[l,p]*abs2(Imax[p]) for p in 1:NC));
    @constraint(TPCSMP, sum(Y[l,p] for p in 1:NC) == 1);
end
for h = 1:H
    @constraint(TPCSMP, V[3*slack-2,h] == 1.0 + im*0.0);
    @constraint(TPCSMP, V[3*slack-1,h] == -0.5 - im*0.866025403784439);
    @constraint(TPCSMP, V[3*slack,h] == -0.5 + im*0.866025403784439);
    for l = 1:NL
        @constraint(TPCSMP,Vj[3*l-2:3*l,h] == branch_data.Lij[l]*
        sum(Y[l,p]*Zp3[3*p-2:3*p,:] for p in 1:NC)*J[3*l-2:3*l,h]);  
        @constraint(TPCSMP, Vj[3*l-2:3*l,h] == 
        sum(A3[3*k-2:3*k,3*l-2:3*l]*V[3*k-2:3*k,h] for k in 1:NN));  
        for j = 0:2
            @constraint(TPCSMP,  abs2(J[3*l-j,h]) - Ilmax[l] <= 0);
        end
    end
    for k = 1:NN
        @constraint(TPCSMP, Ig[3*k-2:3*k,h] - Idy[3*k-2:3*k,h] - 
        transpose(M)*Idd[3*k-2:3*k,h] == 
        sum(A3[3*k-2:3*k,3*l-2:3*l]*J[3*l-2:3*l,h] for l in 1:NL));
        @constraint(TPCSMP, conj(Sg[3*k-2:3*k,h]) == 
        LinearAlgebra.diagm(conj(V[3*k-2:3*k,h]))*Ig[3*k-2:3*k,h]);
        @constraint(TPCSMP, conj(Sdy[3*k-2:3*k])*daily_data.CD[h] ==
        LinearAlgebra.diagm(conj(V[3*k-2:3*k,h]))*Idy[3*k-2:3*k,h]);
        @constraint(TPCSMP, conj(Sdd[3*k-2:3*k])*daily_data.CD[h] ==
        LinearAlgebra.diagm(conj(M*V[3*k-2:3*k,h]))*Idd[3*k-2:3*k,h]);
        for j = 0:2
            @constraint(TPCSMP, abs2(Vmin) <= abs2(V[3*k-j,h]) <= abs2(Vmax));
        end
        if k != slack 
            @constraint(TPCSMP, Sg[3*k-2:3*k,h] == 0);
        end
    end
end
@constraint(TPCSMP, Zloss == (CkWh*T*Sb*dh)*
sum(real(transpose(Vj[:,h])*conj(J[:,h])) for h in 1:H));
@constraint(TPCSMP, Zinv == 3*sum(sum(Y[l,p]*conductor_data.Cinv[p]*
branch_data.Lij[l] for p in 1:NC) for l in 1:NL));
@objective(TPCSMP,Min,Zloss + Zinv);
JuMP.optimize!(TPCSMP);
@show objective_value(TPCSMP);