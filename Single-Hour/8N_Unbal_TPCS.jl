using DataFrames, LinearAlgebra
Vb = 13.8; Sb = 1000; Ib = Sb/Vb; Zb = ((1000*Vb)^2)/(Sb*1000); 
branch_data = DataFrames.DataFrame([
    (1, 2, 1, 1), (2, 3, 1, 1),
    (1, 4, 1, 1), (1, 5, 1, 1),
    (5, 6, 1, 1), (3, 7, 1, 1),
    (3, 8, 1, 1),
]); 
DataFrames.rename!(branch_data, [:i, :j, :Zij, :Lij]);
node_data = DataFrames.DataFrame([
    (1, 0,       0, 0,       0,   0,       0,   0),
    (2, 3162.6,  0,	0,       0,	  0,       0,   0),
    (3, 0,	     0,	2419.5,	 0,	  0,       0,   0),
    (4, 0,	     0,	0,	     0,   7897.5,  0,   0),
    (5, 913.5,	 0,	913.5,	 0,	  0,	   0,   0),
    (6, 0,   	 0,	3051.6,	 0,	  3051.6,  0,   0),
    (7, 2798.4,  0,	0,       0,	  0,       0,   0),
    (8, 1298.55, 0, 2597.1,	 0,	  1298.55, 0,   0),
]);
DataFrames.rename!(node_data, [:i, :Pai, :Qai, :Pbi, :Qbi, :Pci, :Qci, :Type])
NN = size(node_data,1); NL = size(branch_data,1); A3 = zeros(3*NN,3*NL);
for l = 1:NL
    Ni = branch_data.i[l]; Nj = branch_data.j[l];
    A3[3*Ni-2:3*Ni,3*l-2:3*l] = [1 0 0; 0 1 0; 0 0 1];
    A3[3*Nj-2:3*Nj,3*l-2:3*l] = [-1 0 0; 0 -1 0; 0 0 -1];
end
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

conductor_data = DataFrames.DataFrame([
    (1, 180, 1986), (2, 200, 2790),
    (3, 230, 3815), (4, 270, 5090),
    (5, 300, 8067), (6, 340, 12673),
    (7, 600, 23419), (8, 720, 30070),
]);
DataFrames.rename!(conductor_data, [:caliber, :Imax, :Cinv]);
conductor_data.Imax = conductor_data.Imax/Ib;
Z1 = [0.8763 + im*0.4133 0.0    + im*0.0    0.0    + im*0.0;
      0.0    + im*0.0    0.8763 + im*0.4133 0.0    + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.8763 + im*0.4133]; 
Z2 = [0.6960 + im*0.4133 0.0    + im*0.0    0.0    + im*0.0;
      0.0    + im*0.0    0.6960 + im*0.4133 0.0    + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.6960 + im*0.4133];
Z3 = [0.5518 + im*0.4077 0.0    + im*0.0    0.0    + im*0.0;
      0.0    + im*0.0    0.5518 + im*0.4077 0.0    + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.5518 + im*0.4077];
Z4 = [0.4387 + im*0.3983 0.0    + im*0.0    0.0    + im*0.0;
      0.0    + im*0.0    0.4387 + im*0.3983 0.0    + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.4387 + im*0.3983];
Z5 = [0.3480 + im*0.3899 0.0    + im*0.0    0.0 + im*0.0;
      0.0    + im*0.0    0.3480 + im*0.3899 0.0 + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.3480 + im*0.3899];
Z6 = [0.2765 + im*0.3610 0.0    + im*0.0    0.0    + im*0.0;
      0.0    + im*0.0    0.2765 + im*0.3610 0.0    + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.2765 + im*0.3610];
Z7 = [0.0966 + im*0.1201 0.0    + im*0.0    0.0    + im*0.0;
      0.0    + im*0.0    0.0966 + im*0.1201 0.0    + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.0966 + im*0.1201];
Z8 = [0.0853 + im*0.0950 0.0    + im*0.0    0.0    + im*0.0;
      0.0    + im*0.0    0.0853 + im*0.0950 0.0    + im*0.0;
      0.0    + im*0.0    0.0    + im*0.0    0.0853 + im*0.0950];
Zp3 = [Z1;Z2;Z3;Z4;Z5;Z6;Z7;Z8];
Zp3 = Zp3/Zb; NC = size(conductor_data,1);

using JuMP, AmplNLWriter, Bonmin_jll
TPCS = Model(() -> AmplNLWriter.Optimizer(Bonmin_jll.amplexe))
set_attribute(TPCS, "bonmin.nlp_log_level", 0)
set_attribute(TPCS, "honor_original_bounds", "yes")
M = [1 -1 0; 0 1 -1; -1 0 1]; slack = 1;
Vmin = 0.90; Vmax = 1.10;
Imin = 0.0; Imax = conductor_data.Imax;
CkWh = 0.1390; T = 8760;
@variable(TPCS,V[k in 1:3*NN] in ComplexPlane());
for k = 1:NN 
    set_start_value(real(V[3*k-2]),1.0);
    set_start_value(real(V[3*k-1]),-0.5);
    set_start_value(real(V[3*k]),-0.5);
    set_start_value(imag(V[3*k-2]),0.0);
    set_start_value(imag(V[3*k-1]),-0.866025403784439);
    set_start_value(imag(V[3*k]),0.866025403784439);
end
@variable(TPCS, Sg[k in 1:3*NN] in ComplexPlane());
@variable(TPCS, Ig[k in 1:3*NN] in ComplexPlane());
@variable(TPCS, Idy[k in 1:3*NN] in ComplexPlane());
@variable(TPCS, Idd[k in 1:3*NN] in ComplexPlane());
@variable(TPCS, J[l in 1:3*NL] in ComplexPlane());
@variable(TPCS, Vj[l in 1:3*NL] in ComplexPlane());
@variable(TPCS, Ilmax[l in 1:NL]);
@variable(TPCS, Y[l in 1:NL, p in 1:NC], Bin);
@variable(TPCS, Zloss); @variable(TPCS, Zinv);

@constraint(TPCS, V[3*slack-2] == 1.0 + im*0.0);
@constraint(TPCS, V[3*slack-1] == -0.5 - im*0.866025403784439);
@constraint(TPCS, V[3*slack] == -0.5 + im*0.866025403784439);
for l = 1:NL
    @constraint(TPCS, Vj[3*l-2:3*l] == 
    branch_data.Lij[l]*sum(Y[l,p]*Zp3[3*p-2:3*p,:] for p in 1:NC)*J[3*l-2:3*l]);  
    @constraint(TPCS, Vj[3*l-2:3*l] == 
    sum(A3[3*k-2:3*k,3*l-2:3*l]*V[3*k-2:3*k] for k in 1:NN));  
    for j = 0:2
        @constraint(TPCS,  abs2(J[3*l-j]) - Ilmax[l] <= 0);
    end
    @constraint(TPCS, Ilmax[l] == sum(Y[l,p]*abs2(Imax[p]) for p in 1:NC));
    @constraint(TPCS, sum(Y[l,p] for p in 1:NC) == 1);
end
for k = 1:NN
    @constraint(TPCS, Ig[3*k-2:3*k] - Idy[3*k-2:3*k] - 
    transpose(M)*Idd[3*k-2:3*k] == 
    sum(A3[3*k-2:3*k,3*l-2:3*l]*J[3*l-2:3*l] for l in 1:NL));
    @constraint(TPCS, conj(Sg[3*k-2:3*k]) == 
    LinearAlgebra.diagm(conj(V[3*k-2:3*k]))*Ig[3*k-2:3*k]);
    @constraint(TPCS, conj(Sdy[3*k-2:3*k]) == 
    LinearAlgebra.diagm(conj(V[3*k-2:3*k]))*Idy[3*k-2:3*k]);
    @constraint(TPCS, conj(Sdd[3*k-2:3*k]) == 
    LinearAlgebra.diagm(conj(M*V[3*k-2:3*k]))*Idd[3*k-2:3*k]);
    for j = 0:2
        @constraint(TPCS, abs2(Vmin) <= abs2(V[3*k-j]) <= abs2(Vmax));
    end
    if k != slack 
        @constraint(TPCS, Sg[3*k-2:3*k] == 0);
    end
end
@constraint(TPCS, Zloss == (CkWh*T*Sb)*real(transpose(Vj)*conj(J)));
@constraint(TPCS, Zinv == 3*sum(sum(Y[l,p]*conductor_data.Cinv[p]*
branch_data.Lij[l] for p in 1:NC) for l in 1:NL));
@objective(TPCS,Min, Zloss + Zinv);
JuMP.optimize!(TPCS); @show objective_value(TPCS);