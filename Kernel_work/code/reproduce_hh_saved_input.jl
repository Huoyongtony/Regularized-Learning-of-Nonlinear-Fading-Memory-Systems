# Reproduce the HH closed-loop test using the saved thesis input.
# Derived from kernel_learning_new.ipynb; see README.md for provenance.
using LinearAlgebra, DelimitedFiles, Plots
include(joinpath(@__DIR__, "kernel_module.jl"))
include(joinpath(@__DIR__, "signal_system.jl"))
include(joinpath(@__DIR__, "system_kernel.jl"))
using .KernelLearning, .SignalSystem, .SystemKernel
const OUTPUT_DIR = abspath(isempty(ARGS) ? joinpath(@__DIR__, "../../output/hh_saved_input") : ARGS[1])
mkpath(OUTPUT_DIR)


# HH model and Euler channel simulator
  C_m     = 1       # μF/cm², membrane capacitance
  G_NaMax = 120     # mS/cm², maximum conductivity of Na channel
  G_KMax  = 36      # mS/cm², maximum conductivity of K channel
  G_L     = 0.3     # mS/cm², leak conductivity
  V_r  = 65.1     # mV, resting potential
  V_Na =  50      # mV, Nernst voltage for Na 
  V_K  = -77       # mV, Nernst voltage for K
  V_L  = -54.387   # mV, Nernst voltage for leak

alpha_n(dV) = (0.1 - 0.01*(V_r+dV))/(exp(1 - 0.1*(V_r+dV))- 1)
beta_n(dV) = 0.125/(exp(0.0125*(V_r+dV)))
alpha_m(dV) = (2.5 - 0.1*(V_r+dV))/(exp(2.5 - 0.1*(V_r+dV))- 1)
beta_m(dV) = 4/(exp((V_r+dV)/18))
alpha_h(dV) = 0.07/(exp(0.05*(V_r+dV)))
beta_h(dV) = 1/(exp(3 - 0.1*(V_r+dV))+ 1)

n_∞(dV) = alpha_n(dV)/(alpha_n(dV)+beta_n(dV))
m_∞(dV) = alpha_m(dV)/(alpha_m(dV)+beta_m(dV))
h_∞(dV) = alpha_h(dV)/(alpha_h(dV)+beta_h(dV))


function nlsim(sys, u, t ; x0=0)
    dt=t[2]-t[1]
    output_y=Array{Float64,2}(undef,(length(sys(x0,u(x0,t[1]))[2]),length(t)))
    output_x=Array{Float64,2}(undef,(length(x0),length(t)))
    output_u=Array{Float64,2}(undef,(length(u(x0,t[1])),length(t)))
    x=copy(x0)
    for i=1:length(t)
        u_temp=u(x,t[i])
        result=sys(x,u_temp)
        output_y[:,i]=copy(result[2])
        output_x[:,i]=copy(x)
        output_u[:,i]=copy(u_temp)
        x=x+result[1].*dt
    end
    return(output_y,t,output_x,output_u)
end

sys2(x,u)=[-5x.+u,[clamp(0.5*x[1], -5, 5)]]



sys3(x,u)=[(alpha_n(u[1])*(1 .-x).-beta_n(u[1]).*x), 36 .*x.^4*(u[1]-V_K)]

sys4(x,u)=[[(alpha_m(u[1])*(1-x[1])-beta_m(u[1])*x[1]),(alpha_h(u[1])*(1 -x[2])-beta_h(u[1])*x[2])],[120*x[1]^3*x[2]*(u[1]-V_Na)]]

sys5(x,u)=[0,G_L*(V_L-u[1])]
x0=zeros(1);

# Fit potassium channel (1000 training pairs)
num_outputs,input_size,output_size,num_points=(1000,1,1,2000)
num_outputs_temp=50
num_outputs_temp1=30
num_outputs_temp2=20
amplification=1
lambda=0.2
lambda2=0
time_range=20
dt=time_range/num_points
println(dt)
sigma=1
num_point_model=1000
fading_weight_time=10
x=Array{Float64,3}(undef,(num_outputs,input_size,num_point_model))
y=Array{Float64,3}(undef,(num_outputs,output_size,num_point_model))
x_temp=Array{Float64,3}(undef,(num_outputs_temp,input_size,num_points))
y_temp=Array{Float64,3}(undef,(num_outputs_temp,output_size,num_points))
t=range(0,time_range,length=num_points)


t_fading=range(0,fading_weight_time,num_point_model)
fading_weight=Array([exp(lambda*(t_fading[i]-fading_weight_time)) for i=1:num_point_model]');
fading_weight2=Array([exp(lambda2*(t_fading[i]-fading_weight_time)) for i=1:num_point_model]');
fading_weight=fading_weight/(sum(fading_weight.^2)*dt)^0.5*amplification;
fading_weight2=fading_weight2/(sum(fading_weight2.^2)*dt)^0.5*amplification;







for i=1:num_outputs_temp
    bias=-2*i+30+65.1
    tsample,x_temp[i,:,:],y_temp[i,:,:]=Signal_output_step(sys3,
        [[10,time_range*1.5,time_range]], [n_∞(-65.1)], [bias], [1000], [-65.1], 
        num_points,input_size,output_size,1,nlsim)
end


x_temp=x_temp.-V_K

ratio=1/50
start_position=Int(1000)
step=Int((num_points-start_position)*ratio)
Threads.@threads for i=1:num_outputs_temp    
    x[1+(i-1)*step:step+(i-1)*step,:,:]=slice_fading_input(x_temp[i,:,:], ones(size(fading_weight)),ratio,-65.1-V_K,start_position)
    y[1+(i-1)*step:step+(i-1)*step,:,:]=slice_fading_input(y_temp[i,:,:], ones(size(fading_weight)),ratio,0.0,start_position)
end



x_train=Array{Float64,3}(undef,(num_outputs,input_size,num_point_model))
for i in range(1,size(x)[1])
    x_train[i,:,:]=x[i,:,:].*fading_weight
end

y_train=y[:,1,size(y)[3]]



alpha=1
beta=1
Sigma=10
Sigma2=10

gain_schedule_kernel=gain_schedule_kernel_gen4_v3_7(Sigma,dt,num_point_model)
kernel_K=gain_schedule_kernel_gen4_v3_7(Sigma,dt,num_point_model)





K_gain=kernel_matrix(x_train,gain_schedule_kernel)

weight_gain=ridge_regression_kernel(K_gain, y_train, 0.0001)

error_gain=sum(abs.(y_train-K_gain*weight_gain))/size(y)[1]

println(error_gain)
println((weight_gain'*K_gain*weight_gain)^0.5*(sum(fading_weight.^2)*dt)^0.5)

x_K=copy(x)
y_K=copy(y)
x_train_K=copy(x_train)
y_train_K=copy(y_train)
weight_K=copy(weight_gain)
fading_weight_K=copy(fading_weight)
time_range_K=copy(time_range);
num_outputs_K=copy(num_outputs)
input_size_K=copy(input_size)
num_points_K=copy(num_point_model)

function ion_K(input)
    input_fade=Array{Float64,3}(undef,(1,input_size_K,num_points_K))
    input_fade[1,:,:]=(input).*fading_weight_K
    K_star=kernel_matrix(input_fade,x_train_K,kernel_K)
    return K_star*weight_K
end

# Fit sodium channel (2000 training pairs)
num_outputs,input_size,output_size,num_points=(2000,1,1,1500)
num_outputs_temp=40
lambda=0.2
lambda2=1.5
time_range=15
amplification=1
dt=time_range/num_points
println(dt)
sigma=1
num_point_model=1500
fading_weight_time=15
x2=Array{Float64,3}(undef,(num_outputs,input_size,num_point_model))
y2=Array{Float64,3}(undef,(num_outputs,output_size,num_point_model))
x_temp2=Array{Float64,3}(undef,(num_outputs_temp,input_size,num_points))
y_temp2=Array{Float64,3}(undef,(num_outputs_temp,output_size,num_points))
t=range(0,time_range,length=num_points)


t_fading=range(0,fading_weight_time,num_point_model)
fading_weight=Array([exp(lambda*(t_fading[i]-fading_weight_time)) for i=1:num_point_model]');
fading_weight2=Array([exp(lambda2*(t_fading[i]-fading_weight_time)) for i=1:num_point_model]');
fading_weight=fading_weight/(sum(fading_weight.^2)*dt)^0.5*amplification;
fading_weight2=fading_weight2/(sum(fading_weight2.^2)*dt)^0.5*amplification;





sharpness=100
start1=13
end1=13.5
start2=13.8
end2=13.9



for i=0:39
    bias=-2.5*i+20
        tsample,x_temp2[i+1,:,:],y_temp2[i+1,:,:]=Signal_output_step(sys4,
        [[13,13.01,time_range]], [m_∞(-65.1),h_∞(-65.1)], [1000], [100], [bias], 
        num_points,input_size,output_size,1,nlsim)
end




x_temp2=x_temp2 .- V_Na

ratio=1/60
start_position=Int(0)
step=Int((num_points-start_position)*ratio)
Threads.@threads for i=1:num_outputs_temp 
    x2[1+(i-1)*step:step+(i-1)*step,:,:]=slice_fading_input(x_temp2[i,:,:], ones(size(fading_weight)),ratio,-65.1-V_Na,start_position)
    y2[1+(i-1)*step:step+(i-1)*step,:,:]=slice_fading_input(y_temp2[i,:,:], ones(size(fading_weight)),ratio,0.0,start_position)
end

ratio=1/8
start_position=Int(1300)
step=Int((num_points-start_position)*ratio)
Threads.@threads for i=1:num_outputs_temp 
    x2[1001+(i-1)*step:1000+step+(i-1)*step,:,:]=slice_fading_input(x_temp2[i,:,:], ones(size(fading_weight)),ratio,-65.1-V_Na,start_position)
    y2[1001+(i-1)*step:1000+step+(i-1)*step,:,:]=slice_fading_input(y_temp2[i,:,:], ones(size(fading_weight)),ratio,0.0,start_position)
end



x_train2=Array{Float64,3}(undef,(num_outputs,input_size,num_point_model))
for i in range(1,size(x2)[1])
    x_train2[i,:,:]=x2[i,:,:].*fading_weight
end

y_train2=y2[:,1,size(y2)[3]]



alpha=1
beta=0
Sigma=9.5
Sigma2=2.5
Sigma3=1

gain_schedule_kernel=gain_schedule_kernel_gen4_v3_2(Sigma,Sigma2,dt,num_point_model,fading_weight2)
kernel_Na=gain_schedule_kernel_gen4_v3_2(Sigma,Sigma2,dt,num_point_model,fading_weight2)







K_gain2=kernel_matrix(x_train2,gain_schedule_kernel)


weight_gain2=ridge_regression_kernel(K_gain2, y_train2, 0.0000001)






error=sum(abs.(y_train2-K_gain2*weight_gain2))/size(y2)[1]

println(error)
println((weight_gain2'*K_gain2*weight_gain2)^0.5*(sum(fading_weight.^2)*dt)^0.5)

time_range_Na=copy(time_range)
fading_weight_Na=copy(fading_weight)
x_Na=copy(x2)
y_Na=copy(y2)
x_train_Na=copy(x_train2)
y_train_Na=copy(y_train2)
weight_Na=copy(weight_gain2);
num_outputs_Na=copy(num_outputs)
input_size_Na=copy(input_size)
num_points_Na=copy(num_point_model)

function ion_Na(input)
    input_fade=Array{Float64,3}(undef,(1,input_size_Na,num_points_Na))
    input_fade[1,:,:]=(input).*fading_weight_Na
    K_star=kernel_matrix(input_fade,x_train_Na,kernel_Na)
    return K_star*weight_Na
end

# Use the original saved realization, never a new random draw.
input = vec(readdlm(joinpath(@__DIR__, "test2_input.txt"), Float64))
@assert length(input) == 4000 && all(isfinite, input)


# Learned closed-loop model
output_5=ones(1,4000)*(-65.1)
output_K4=zeros(1,4000)
output_Na4=zeros(1,4000)
rest_potential=-65.1

x=[-65.1,0,0] #v,m,h,n


dt=0.01

for i=1:3999

    if num_points_Na<num_points_K
        if i<num_points_K
            v_past=[ones(1,num_points_K - i)*rest_potential copy(output_5[:,1:i])]
        else
            v_past=output_5[:,i - length(fading_weight_K) + 1:i]
        end
        v_past_Na=v_past[:,1+(num_points_K-num_points_Na):num_points_K]
    else
        if i<num_points_Na
            v_past_Na=[ones(1,num_points_Na - i)*rest_potential copy(output_5[:,1:i])]
        else
            v_past_Na=output_5[:,i - length(fading_weight_Na) + 1:i]
        end
        v_past=v_past_Na[:,1+(num_points_Na-num_points_K):num_points_Na]
    end


    result_Na=ion_Na(v_past_Na.-V_Na)
    result_K=ion_K(v_past.-V_K)
    
    dVm = (input[i] - result_Na[1] - result_K[1]  - (x[1]-V_L)*G_L)/C_m 

    x[1]=x[1]+dVm*dt


    output_5[1,i+1]=x[1]
    

    output_K4[1,i+1]=result_K[1]
    output_Na4[1,i+1]=result_Na[1]
end

plot_Na_K=plot([output_5' output_K4' output_Na4'])
savefig(plot_Na_K, joinpath(OUTPUT_DIR, "HH_model_appK_app_Na_test2.png"))
plot_Na_K

# Original HH model (preserves historical pre-update recording)
output_noise=[-65.1]
output_K_noise=[0.0]
output_Na_noise=[0.0]
x=[-65.1,m_∞(-65.1),h_∞(-65.1),n_∞(-65.1)] #v,m,h,n

dt=0.01

for i=1:3999
      
    result_Na=sys4(x[2:3],[x[1]])
    result_K=sys3(x[4],[x[1]])
    
    dVm = (input[i] -result_Na[2][1] - result_K[2][1]  - (x[1]-V_L)*G_L)/C_m 
    
    push!(output_noise,x[1])
    x[1]=x[1]+dVm*dt
    x[2:3]=x[2:3]+result_Na[1]*dt
    x[4]=x[4]+result_K[1]*dt
   
    push!(output_K_noise,result_K[2][1])
    push!(output_Na_noise,result_Na[2][1])
end

plot_noise=plot([output_noise output_K_noise output_Na_noise])

# Thesis plots
t_range = LinRange(0, 40, 4000)
plot_test2_summary=plot(t_range,[output_noise output_5'],label=["original" "approx_K_Na" ],xlabel="time/s", ylabel="mV")
plot_test2_input=plot(t_range,input,xlabel="time/s", ylabel="Current/mA")

savefig(plot_test2_summary, joinpath(OUTPUT_DIR, "HH_model_test2_summary.png"))
savefig(plot_test2_input, joinpath(OUTPUT_DIR, "HH_model_test2_summary_input.png"))

writedlm(joinpath(OUTPUT_DIR, "closed_loop_traces.txt"), hcat(t_range, input, output_noise, vec(output_5)))

plot_test2_summary

println("HH figures and traces written to ", OUTPUT_DIR)
