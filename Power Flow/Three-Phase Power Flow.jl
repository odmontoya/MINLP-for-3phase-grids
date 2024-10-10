using DataFrames, LinearAlgebra
function Matrix4N(C)
if C == 1
    Zm = [2.34876 + im*0.895068  0.06348 + im*0.431664  0.06348 + im*0.431664
          0.06348 + im*0.431664  2.34876 + im*0.895068  0.06348 + im*0.431664
          0.06348 + im*0.431664  0.06348 + im*0.431664  2.34876 + im*0.895068];
end
return Zm;
end
Vb = 13.8/sqrt(3); Sb = 1000; Zb = ((1000*Vb)^2)/(Sb*1000); 
branch_data = DataFrames.DataFrame([
    (1,	2, 1, 5280),
    (2,	3, 1, 5280),
    (2,	4, 1, 5280),
]); 
DataFrames.rename!(branch_data, [:i, :j, :Zij, :Lij]);
branch_data.Lij = branch_data.Lij*0.000189394;
node_data = DataFrames.DataFrame([
    (1, 0,   0,   0,   0,   0,   0,   0),
    (2, 300, 200, 100, 230, 300, 100, 0),
    (3, 260, 20,  20,  100, 300, 20,  0),
    (4, 210, 50,  210, 50,  210, 50,  0),
]);
DataFrames.rename!(node_data, [:i, :Pai, :Qai, :Pbi, :Qbi, :Pci, :Qci, :Type])
NN = size(node_data,1); NL = size(branch_data,1);
A3 = zeros(3*NN,3*NL); Zp3 = complex(zeros(3*NL,3*NL));
for l = 1:NL
    Ni = branch_data.i[l]; Nj = branch_data.j[l];
    A3[3*Ni-2:3*Ni,3*l-2:3*l] = [1 0 0; 0 1 0; 0 0 1];
    A3[3*Nj-2:3*Nj,3*l-2:3*l] = [-1 0 0; 0 -1 0; 0 0 -1];
    local Zp3[3*l-2:3*l,3*l-2:3*l] = 
    (Matrix4N(branch_data.Zij[l])*branch_data.Lij[l])/Zb;
end
Sdy = complex(zeros(3*(NN),1));
Sdd = complex(zeros(3*(NN),1));
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
using JuMP, Ipopt
TPPF = Model(Ipopt.Optimizer)
slack = 1; M = [1 -1 0; 0 1 -1; -1 0 1];
@variable(TPPF,V[k in 1:3*NN] in ComplexPlane());
for k = 1:NN 
    set_start_value(real(V[3*k-2]),1.0); 
    set_start_value(imag(V[3*k-2]),0.0);
    set_start_value(real(V[3*k-1]),-0.5); 
    set_start_value(imag(V[3*k-1]),-0.866025403784439);
    set_start_value(real(V[3*k]),-0.5); 
    set_start_value(imag(V[3*k]),0.866025403784439);
end
@variable(TPPF, Sg[k in 1:3*NN] in ComplexPlane());
@variable(TPPF, Ig[k in 1:3*NN] in ComplexPlane());
@variable(TPPF, Idy[k in 1:3*NN] in ComplexPlane());
@variable(TPPF, Idd[k in 1:3*NN] in ComplexPlane());
@variable(TPPF, Ij[l in 1:3*NL] in ComplexPlane());
@variable(TPPF, Vj[l in 1:3*NL] in ComplexPlane());
@constraint(TPPF, V[3*slack-2] == 1.0 + im*0.0);
@constraint(TPPF, V[3*slack-1] == -0.5 - im*0.866025403784439);
@constraint(TPPF, V[3*slack] == -0.5 + im*0.866025403784439);
for l = 1:NL
    @constraint(TPPF, Vj[3*l-2:3*l] == Zp3[3*l-2:3*l,3*l-2:3*l]*Ij[3*l-2:3*l]);  
    @constraint(TPPF, Vj[3*l-2:3*l] == 
    sum(A3[3*k-2:3*k,3*l-2:3*l]*V[3*k-2:3*k] for k in 1:NN));  
end    
for k = 1:NN
    @constraint(TPPF, Ig[3*k-2:3*k] - Idy[3*k-2:3*k] - 
    transpose(M)*Idd[3*k-2:3*k] == 
    sum(A3[3*k-2:3*k,3*l-2:3*l]*Ij[3*l-2:3*l] for l in 1:NL));
    @constraint(TPPF, conj(Sg[3*k-2:3*k]) == 
    LinearAlgebra.diagm(conj(V[3*k-2:3*k]))*Ig[3*k-2:3*k]);
    @constraint(TPPF, conj(Sdy[3*k-2:3*k]) == 
    LinearAlgebra.diagm(conj(V[3*k-2:3*k]))*Idy[3*k-2:3*k]);
    @constraint(TPPF, conj(Sdd[3*k-2:3*k]) == 
    LinearAlgebra.diagm(conj(M*V[3*k-2:3*k]))*Idd[3*k-2:3*k]);
    if k != slack 
        @constraint(TPPF, Sg[3*k-2:3*k] == 0);
    end
end
@objective(TPPF,Min,Sb*real(transpose(Vj)*conj(Ij)));
JuMP.optimize!(TPPF); @show objective_value(TPPF);