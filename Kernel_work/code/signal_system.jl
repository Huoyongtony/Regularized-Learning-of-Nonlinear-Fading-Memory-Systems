module SignalSystem

using ControlSystems
using ControlSystemsBase: nonlinearity
using TensorCast
using Base.Threads

export Simple_Nonlinear_sys_gen,Signal_output_step,slice_fading_input,Signal_output_sin,Simple_Nonlinear_sys_gen2
export basis_fun, basis_fun2, map2Rn, scattering_trans,Signal_output
export unit_norm_conditioner,passive_conditioner

function Simple_Nonlinear_sys_gen(max,gradient,zeros,poles)
    nl   = nonlinearity(x->clamp(x, -max, max))
    return nl*tf(zpk(zeros,poles, gradient))
end

function Simple_Nonlinear_sys_gen2(max,gradient,zeros,poles,constant)
    nl   = nonlinearity(x->clamp(x, -max, max))
    return nl*tf(zpk(zeros,poles, gradient))+tf(1)*constant
end

function Signal_output_step(sys, time_range, num_points, x0, amp, sharpness,solver=lsim)
    @assert time_range[2] > time_range[1] "Invalid time range"
    @assert amp >= 0 "amp must be greater than or equal to 0"
    t = range(0, stop=time_range[3], length=num_points)
    u(x,t) = [1 / (exp(-sharpness * (t - time_range[1])) + 1) * amp * 1 / (exp(-sharpness * (-t + time_range[2])) + 1) ]
    y, _, x, uout = solver(sys, u, t, x0=x0)
    return t, uout, y
end

function Signal_output(sys, time_range, x0, u, num_points,input_size,output_size,solver=lsim,mode="continous")

    outputx =Array{Float64,3}(undef,(1,input_size,num_points))
    outputy = Array{Float64,3}(undef,(1,output_size,num_points))
    t = range(0, stop=time_range, length=num_points)

    y, _, x, uout = solver(sys, u, t, x0=x0)
    outputx[1,:,:] = uout
    outputy[1,:,:] = y
    
    return t, outputx, outputy
end

function Signal_output_step(sys, time_range, x0, amp,sharpness, bias, num_points,input_size,output_size,num_outputs,solver=lsim,mode="continous")
    outputx =Array{Float64,3}(undef,(num_outputs,input_size,num_points))
    outputy = Array{Float64,3}(undef,(num_outputs,output_size,num_points))
    t = range(0, stop=time_range[1][3], length=num_points)
    step = amp ./ num_outputs
    if mode=="continous"
        Threads.@threads for i in 1:num_outputs
            amp_val = i  .* step
            u(x,t) = [1 / (exp(-sharpness[j] * (t - time_range[j][1])) + 1) * amp_val[j] * 
                1 / (exp(-sharpness[j] * (-t + time_range[j][2])) + 1)+bias[j] for j in 1:input_size]
            y, _, x, uout = solver(sys, u, t, x0=x0)
            outputx[i,:,:] = uout
            outputy[i,:,:] = y
        end
    else
        Threads.@threads for i in 1:num_outputs
            amp_val = i .* step
            u(x,t) = [(t>=time_range[j][1])*(t<=time_range[j][2])*amp_val[j]+bias[j] for j in 1:input_size]
            y, _, x, uout = solver(sys, u, t, x0=x0)
            outputx[i,:,:] = uout
            outputy[i,:,:] = y
        end
    end
    return t, outputx, outputy
end

function Signal_output_sin(sys,time_range, x0, fre, amp, shift, bias, num_points,input_size,output_size,num_outputs,solver=lsim)
    outputx =Array{Float64,3}(undef,(num_outputs,input_size,num_points))
    outputy = Array{Float64,3}(undef,(num_outputs,output_size,num_points))
    t=range(0,time_range,length=num_points)
    step = fre ./ num_outputs
    for i in 1:num_outputs
        fre_val = (i) .* step
        u(x,t) = [amp[j]*sin(fre_val[j]*t+shift[j]) + bias[j] for j in 1:input_size]
        y, _, x, uout = solver(sys,u,t,x0=x0);
        outputy[i,:,:]=y
        outputx[i,:,:]=uout
    end;
  
    return t,outputx,outputy
end;


function slice_fading_input(x::Array{T,2}, fading_weight::Array{T},rate,filled=0,start_position=0) where {T<:Real}
    ratio=Int(1/rate)
    demo_ = Array{Float64,3}(undef, (div(size(x)[2]-start_position, ratio),size(x)[1],length(fading_weight)))
    for i = 1:div(size(x)[2]-start_position, ratio)
        if length(fading_weight) > i * ratio+start_position
            demo_[i,:,:] = [ones(size(x)[1],length(fading_weight) - i * ratio-start_position).*filled copy(x[:,1:start_position+i * ratio])] .* fading_weight
        else
            demo_[i,:,:] = x[:,i * ratio - length(fading_weight) + 1+start_position:i * ratio + start_position] .* fading_weight
        end
    end
    return demo_
end

function slice_fading_input(x::Array{T,3}, fading_weight::Array{T},rate,filled=0,start_position=0) where {T<:Real}
    ratio=Int(1/rate)
    demo_ = Array{Float64,4}(undef, (size(x)[1],div(size(x)[3], ratio),size(x)[2],length(fading_weight)))
    for i = 1:size(x)[1]
        demo_[i,:,:,:] =copy(slice_fading_input(x[i,:,:],fading_weight,rate,filled,start_position))
    end
    return demo_
end

function basis_fun(order,time,l,lambda)
    basis = Array{ComplexF64,2}(undef, (2*order+1,l))
    @threads for ome = -order:order
        signal = [exp(-(time-x)/2*lambda)*exp(im*2*pi*ome*exp((-time+x)*lambda)) for x in range(0, time, length=l)]
        signal /= norm(signal)
        basis[ome+order+1,:] = signal
    end
    basis[2*order+1,:]=[zeros(l-1); 1]
    return basis
end;

function basis_fun2(order,time,l,lambda)
    basis =Array{ComplexF64,2}(undef, (2*order+2,l))
    signal = [exp(-(time-x)/2*lambda)*cos(2*pi*0*exp((-time+x)*lambda)) for x in range(0, time, length=l)]
    signal /= norm(signal)
    basis[1,:]=signal
    @threads for ome = 1:order
        signal = [exp(-(time-x)/2*lambda)*cos(2*pi*ome*exp((-time+x)*lambda)) for x in range(0, time, length=l)]
        signal /= norm(signal)
        signal2 = [exp(-(time-x)/2*lambda)*sin(2*pi*ome*exp((-time+x)*lambda)) for x in range(0, time, length=l)]
        signal2 /= norm(signal2)
        basis[(ome)*2,:]=signal
        basis[(ome)*2+1,:]=signal2
    end

    basis[2*order+2,:] = [zeros(l-1); 1]
    
    return basis
end;

function map2Rn(basis, samples)
    output_ = Array{Float64,3}(undef, (size(samples)[1],size(samples)[2],size(basis)[1]))
    @threads for i = 1:size(samples)[1]
        output_[i,:,:] = (samples[i,:,:] * basis')
    end
    newdim=size(samples)[2]*size(basis)[1]
    output = Array{Float64,2}(undef, (size(samples)[1],newdim))
    
    @threads for i = 1:size(samples)[1]
        output[i,:] = reshape(output_[i,:,:], (newdim))
    end

    return output
end

function scattering_trans(input,output)
    input_trans=0.5.*(input+output)
    output_trans=0.5.*(input-output)
    return input_trans,output_trans
end;

function scattering_trans_I(input,output)
    input_trans=0.5.*(input+output)
    output_trans=0.5.*(input-output)
    return input_trans,output_trans-input_trans
end;


function unit_norm_conditioner(input,output,dt)
    norm1=(sum((input).^2)*dt)^0.5
    norm2=sum((output).^2)^0.5
    return min(norm1,norm2)*output/sum(output.^2)^0.5
end;

function passive_conditioner(input,output,dt)
    l=size(input)[2]
    inner_product=max(0,sum(input.*output)*dt)
    norm_output=sum(output[:,l].^2)^0.5
    norm_input=sum(input[:,l].^2)^0.5
    return (output[:,l]/norm_output*(1-exp(-inner_product*10))+input[:,l]/norm_input*exp(-inner_product*10))*norm_output*(1-exp(-inner_product*1000))
end;

end