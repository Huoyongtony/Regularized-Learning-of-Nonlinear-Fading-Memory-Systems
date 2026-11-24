module SystemKernel

include("./kernel_module.jl")
include("./signal_system.jl")

using LinearAlgebra
using ControlSystems
using OrdinaryDiffEq
using TensorCast
using Random: seed!
using .KernelLearning
using .SignalSystem

export rbf_kernel_signal_gen,linear_kernel_signal_gen,sim_sys,exp_kernel_signal_gen,rbf_w_kernel_signal_gen,
rbf_w_kernel_signal_gradient_dw,max_kernel_w_signal_gen,rbf_simulation_invers,
rbf_I_kernel_signal_gen,cone_kernel_signal_gen2,cone_kernel_signal_gen,
cone_kernel_signal_gen3,cone_kernel_signal_gen4,gain_schedule_kernel_gen,gain_schedule_kernel_gen2,gain_schedule_kernel_gen3,
gain_schedule_kernel_gen5,gain_schedule_kernel_gen4,gain_schedule_kernel_max_gen,gain_schedule_kernel_gen4_v2,
gain_schedule_kernel_gen4_v3,gain_schedule_kernel_gen4_v4,gain_schedule_kernel_gen4_v5,gain_schedule_kernel_gen4_v6,
gain_schedule_kernel_gen4_v7,gain_schedule_kernel_gen4_v8,gain_schedule_kernel_gen4_v9,gain_schedule_kernel_gen4_v10,
gain_schedule_kernel_gen4_v11,gain_schedule_kernel_gen4_v3_2,gain_schedule_kernel_gen4_v3_3,
gain_schedule_kernel_gen4_v3_4,gain_schedule_kernel_gen4_v3_5,
gain_schedule_kernel_gen4_v3_6,gain_schedule_kernel_gen6, gain_schedule_kernel_gen7,gain_schedule_kernel_gen4_v3_7,gain_schedule_kernel_gen_v2,gain_schedule_kernel_gen_v3,gain_schedule_kernel_gen4_v3_8
function rbf_kernel_signal_gen(σ,dt)
    function rbf_kernel(x, y)
        return exp(-sum((x.-y).^2)*dt / (2σ^2))
    end
    return rbf_kernel
end

function cone_kernel_signal_gen(a,b,dt)
    function rbf_kernel(x, y)
        norm=(sum((x.-y).^2)*dt)^0.5
        return max(0,(b-a*norm))
    end
    return rbf_kernel
end

function cone_kernel_signal_gen2(a,b,dt)
    function rbf_kernel(x, y)
        norm=(sum((x.-y).^2)*dt)
        return max(exp(-norm/2),max(1-a*norm,1-b*norm^0.5))
    end
    return rbf_kernel
end

function cone_kernel_signal_gen3(a,b,dt)
    function rbf_kernel(x, y)
        norm=(sum((x.-y).^2)*dt)^0.5
        return max(exp(-norm^2/2),max(1-a*norm^3,1-b*norm))
    end
    return rbf_kernel
end

function cone_kernel_signal_gen4(a,b,dt)
    point=((1-b)/a)^(1/3)
    gradient=-3*a*point^2
    function rbf_kernel(x, y)
        norm=(sum((x.-y).^2)*dt)^0.5
        return max(0,min(max(1-a*norm^3,b),gradient*(norm-point)+b))
    end
    return rbf_kernel
end

function gain_schedule_kernel_gen(σ,dt)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(norm1-norm2)^2/(2σ^2))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen_v3(σ,dt)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(norm1-norm2)^2/(2σ^2))*(sum((x.*y))*dt+1)
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen_v2(σ,dt,a)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x.^2)*dt)^0.5
        norm2=(sum(y.^2)*dt)^0.5
        return exp(-(norm1-norm2)^2/(2σ^2))*(sum((x.*y))*dt+a)
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_max_gen(a,dt)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x.^2)*dt)^0.5
        norm2=(sum(y.^2)*dt)^0.5
        norm=abs(norm1-norm2)
        return max(0,(1-a*norm))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen2(σ,dt,l)
    function gain_schedule_kernel(x, y)
        return exp(-sum((x[:,l].-y[:,l]).^2)/(2σ^2))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end



function gain_schedule_kernel_gen3(σ,dt,l)
    function gain_schedule_kernel(x, y)
        return (1+exp(-sum((x[:,l].-y[:,l]).^2)/(2σ^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4(σ1,σ2,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x.^2)*dt)^0.5
        norm2=(sum(y.^2)*dt)^0.5
        return (a*exp(-sum((x[:,l].-y[:,l]).^2)/(2σ1^2))*b*exp(-(norm1-norm2)^2/(2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v2(σ1,σ2,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x.^2)*dt)^0.5
        norm2=(sum(y.^2)*dt)^0.5
        return (a*exp(-sum((x[:,l].-y[:,l]).^2)/(2σ1^2))+b*exp(-(norm1-norm2)^2/(2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3(σ1,σ2,dt,l)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return (exp(-sum((x[:,l].-y[:,l]).^2)/(2σ1^2))*exp(-(norm1-norm2)^2/(2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3_2(σ1,σ2,dt,l,w)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        norm3=(sum(x.*w)*dt)
        norm4=(sum(y.*w)*dt)
        return (exp(-sum(norm1-norm2)^2/(2σ1^2))*exp(-(norm3-norm4)^2/(2σ2^2)))*sum((x.*y))*dt*sum(x[:,l]*y[:,l]')
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3_3(σ1,σ2,dt,l,w)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        norm3=(sum(x.*w)*dt)
        norm4=(sum(y.*w)*dt)
        return (exp(-sum(norm1-norm2)^2/(2σ1^2))*exp(-(norm3-norm4)^2/(2σ2^2)))*sum(x[:,l]*y[:,l]')
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3_4(σ2,dt,l,fade_weight)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(norm1-norm2)^2/(2σ2^2))*sum((x.*y.*fade_weight))*dt*sum(x[:,l]*y[:,l]')
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3_5(σ2,dt,l)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(norm1-norm2)^2/(2σ2^2))*sum(x[:,l]*y[:,l]')
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3_6(σ2,dt,l,c)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(norm1-norm2)^2/(2σ2^2))*(sum(x[:,l]*y[:,l]')+c)^2
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3_7(σ2,dt,l)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(norm1-norm2)^2/(2σ2^2))*sum((x.*y))*dt*sum(x[:,l]*y[:,l]')
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v3_8(σ2,dt,l,alpha)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(norm1-norm2)^2/(2σ2^2))*(sum((x.*y))*dt*+alpha)*sum(x[:,l]*y[:,l]')
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v4(σ1,σ2,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return (a*exp(-sum((x[:,l].-y[:,l]).^2)/(2σ1^2))+b*exp(-(norm1-norm2)^2/(2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v5(σ,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(a*sum((x[:,l].-y[:,l]).^2)+b*(norm1-norm2)^2)/(2σ^2))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v6(σ,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return exp(-(a*sum((x[:,l].-y[:,l]).^2)*b*(norm1-norm2)^2)/(2σ^2))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v7(σ1,σ2,σ3,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return a*(exp(-sum((x[:,l].-y[:,l]).^2)/(2σ1^2))*exp(-(norm1-norm2)^2/(2σ2^2)))*sum((x.*y))*dt+b*exp(-sum((x.-y).^2)*dt / (2σ3^2))
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v8(σ1,σ2,σ3,dt,l,a,b,c)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return (a*exp(-sum((x[:,l].-y[:,l]).^2)/(2σ1^2))+b*exp(-(norm1-norm2)^2/(2σ2^2)))*sum((x.*y))*dt+c*exp(-sum((x.-y).^2)*dt / (2σ3^2))
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v9(σ,σ3,dt,l,a,b,c)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return (exp(-(a*sum((x[:,l].-y[:,l]).^2)+b*(norm1-norm2)^2)/(2σ^2)))*sum((x.*y))*dt+c*exp(-sum((x.-y).^2)*dt / (2σ3^2))
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v10(σ,σ2,dt,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        norm3=sum((x/(norm(x)+1e-8).-y/(norm(y)+1e-8)).^2)*dt
        return (a*exp(-((norm1-norm2)^2)/(2σ^2))*b*exp(-norm3 / (2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v11(σ,σ2,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        diff1=(x[:,2:l]-x[:,1:l-1])
        diff2=(y[:,2:l]-y[:,1:l-1])
        diff=sum((diff1-diff2).^2)*dt
        return (a*exp(-((norm1-norm2)^2)/(2σ^2))*b*exp(-(diff)/(2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen4_v12(σ,σ2,dt,l,a,b)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return (a*exp(-((norm1-norm2)^2)/(2σ^2))*b*exp(-(diff)^2/(2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen5(σ,dt,l)
    function gain_schedule_kernel(x, y)
        return (1+exp(-sum((x[:,l].-y[:,l]).^2)/(2σ^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen6(σ,σ2,dt,l)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return (1+exp(-sum((x[:,l].-y[:,l]).^2)/(2σ^2))*exp(-((norm1-norm2)^2)/(2σ2^2)))*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end

function gain_schedule_kernel_gen7(σ,dt,l,a)
    function gain_schedule_kernel(x, y)
        norm1=(sum(x)*dt)
        norm2=(sum(y)*dt)
        return (1+a*exp(-((norm1-norm2)^2)/(2σ^2)))*sum(x[:,l].*y[:,l])*sum((x.*y))*dt
    end
    return gain_schedule_kernel
end


function rbf_I_kernel_signal_gen(σ,dt,num_points,alpha,beta)
    function rbf_kernel(x, y)
        return exp(-(sum((x.-y).^2)*dt*(alpha)+sum((x[:,num_points].-y[:,num_points]).^2)*beta) / (2σ^2))
    end
    return rbf_kernel
end

function linear_kernel_signal_gen(dt)
    function linear_kernel(x, y)
        return sum((x.*y))*dt
    end
    return linear_kernel
end

function rbf_simulation_invers(x_train,weight,x,fading_weight,σ,dt,it,err,mode="rbf",alpha=0.5,beta=0.5)

    rbf_kernel_O(a,b)=exp( -(a+b)*dt / (2σ^2))
    rbf_kernel_I(a,b)=exp( -((a+b)*dt*(alpha)+b*beta) / (2σ^2))

    function cone_kernel(a, b)
        norm=((a+b)*dt)^0.5
        return  max(0,(beta-alpha*norm))
    end

    function cone_kernel2(a, b)
        norm=((a+b)*dt)
        return max(exp(-norm^2/2),max(1-alpha*norm,1-beta*norm^0.5))
    end

    function cone_kernel3(a, b)
        norm=((a+b)*dt)^0.5
        return max(exp(-norm^2/2),max(1-alpha*norm^3,1-beta*norm))
    end

    function gain_schedule_kernel(a, b)
        return (1+exp(-sum((b).^2)/(2σ^2)))*sum((x.*y))*dt
    end


    if mode=="rbf"
        rbf_kernel=rbf_kernel_O
    end

    if mode=="rbf_I"
        rbf_kernel=rbf_kernel_I
    end

    if mode=="cone"
        rbf_kernel=cone_kernel
    end

    if mode=="cone2"
        rbf_kernel=cone_kernel2
    end

    if mode=="cone3"
        rbf_kernel=cone_kernel3
    end

    loop=zeros(size(x)[1])
    loop_pre=copy(loop)

    length_fading_weight=size(fading_weight)[2]
    
    output=zeros((size(x)[1],size(x)[2]))

    num_x_train=size(x_train)[1]

    num_x_input=size(x_train)[2]

    num_point=size(x_train)[3]

    a=zeros(size(x_train)[1])

    K = zeros(num_x_train, 1)

    b=zeros((num_x_train,size(x_train)[2]))


    for i=1:size(x)[2]

        if 1<i<length_fading_weight
            a[:]=sum((x_train[:,:,length_fading_weight-i+1:length_fading_weight-1].-reshape((x[:,1:i-1]-output[:,1:i-1]).*fading_weight[:,length_fading_weight-i+1:length_fading_weight-1],1,num_x_input,i-1)).^2, dims=(2,3))
            +sum(x_train[:,:,1:length_fading_weight-i].^2,dims=(2,3))
        end

        if i>=length_fading_weight
            a[:]=sum((x_train[:,:,1:length_fading_weight-1].-reshape((x[:,1+i-length_fading_weight:i-1]-output[:,1+i-length_fading_weight:i-1]).*fading_weight[:,1:length_fading_weight-1],1,num_x_input,length_fading_weight-1)).^2,dims=(2,3))
        end


        # if i>1
        #     a_temp=x_train[:,:,1:length_fading_weight-1].-[zeros(size(x)[1],length_fading_weight - i) (x[:,1:i-1])] .* fading_weight[:,1:length_fading_weight-1]
        #     Threads.@threads for j=1:num_x_train
        #         a[j]=sum(a_temp[j,:,:].^2)
        #     end
        # end

        if i==1
            a[:]=sum(x_train[:,:,1:length_fading_weight-1].^2,dims=(2,3))
        end
 
        for j=1:it
            loop_pre=copy(loop)
            b[:]=(x_train[:,:,length_fading_weight].-reshape((x[:,i]-loop).*fading_weight[:,length_fading_weight],1,num_x_input)).^2
  
            Threads.@threads for k=1:num_x_train
                K[k,1]=rbf_kernel(a[k],sum(b[k,:]))
            end

            loop=weight'*K

            if maximum(abs.(loop_pre-loop))<err
                break
            end

        end    

        if i % 100  ==0
            println(maximum(abs.(loop_pre-loop)))
        end
        
        output[:,i]=loop

    end


    return output

end
# function rbf_w_kernel_signal_gen(σ,dt,w)
#     function rbf_kernel(x, y)
#         return exp(-sum((w.*(x-y)).^2)*dt / (2σ^2))
#     end
#     return rbf_kernel

# end    

# function max_kernel_w_signal_gen(dt,w)
#     function max_kernel(x, y)
#         return max(sum(w .*x)*dt, sum(w .*y)*dt)
#     end
#     return max_kernel
# end

# function rbf_w_kernel_signal_gradient_dw(σ,dt,w,x,y)
#     constant=dt / (2σ^2)
#     return -exp(-sum((w.*(x-y)).^2)*constant)*constant*2*(x-y).^2 .*w
# end

function rbf_w_kernel_signal_gen(σ,dt,w)

    function rbf_kernel(x, y)
        return exp(-sum((w.*(x-y))*dt)^2 / (2σ^2))
    end
    return rbf_kernel

end   

function rbf_w_kernel_signal_gradient_dw(σ,dt,w,x,y)

    constant=1 / (2σ^2)
    return -exp(-sum((w.*(x-y))*dt)^2*constant)*constant*2*sum((w.*(x-y))*dt)*(x-y)

end


function exp_kernel_signal_gen(σ,dt)
    dt2=dt^2
    function exp_kernel(x, y)
        return exp(sum((x.*y)).^2*dt2 / (2σ^2))
    end
    return exp_kernel
end

function sim_sys(input,output_size,fading_weight,para,predict_fun,mode="kernel")

    window_size=size(fading_weight)[1]
    num_input=size(input)[1]
    output=Array{Float64,2}(undef, (size(input)[2],output_size))

    if mode=="kernel"

        weight=para.weight
        X=para.X
        k=para.k
        
        for i=1:size(input)[2]
            if i<size(fading_weight)[1]
                output[:,i]=predict_fun(k,weight,X,([zeros(num_input,window_size - i) input[:,1:i]].*fading_weight))
            else
                output[:,i]=predict_fun(k,weight,X,(input[:,window_size-i+1:i].*fading_weight))
            end
        end;
    end

    if mode=="random_feature"

        weight=para.weight
        random_feature_para=para.random_feature_para

        for i=1:size(input)[2]
            if i<size(fading_weight)[1]
                output[:,i]=predict_fun(weight,random_feature_para,([zeros(num_input,window_size - i) input[:,1:i]].*fading_weight))
            else
                output[:,i]=predict_fun(weight,random_feature_para,(input[:,window_size-i+1:i].*fading_weight))
            end
        end;

    end
    
    
    return output
end;




end