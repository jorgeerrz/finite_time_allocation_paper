module accumulator_functions

using PyPlot, Distributions, SpecialFunctions, LaTeXStrings, LambertW, CurveFit, LinearAlgebra

export utility_analytical_Gaussian
export utility_simulations_Gaussian
export utility_analytical_flat
export utility_simulations_flat
export utility_simulations_bimodal_Gaussian
export utility_analytical_bimodal
export utility_directions
export compute_even_allocation
export plot_Mopt_vs_C
export plot_utility_even
export plot_utility_triangle
export plot_uniform
export plot_directions
export plot_ascent_random
export deriv_Gaussian
export compute_ascent_CV 
export compute_ascent_distances_symmetric
export ascent

function normal(z,mean,variance)
    exp(-(z-mean)^2/(2*variance))/sqrt(2*pi*variance)
end

function plot_uniform(x)
    out = zeros(length(x))
    for i in 1:length(x)
        if x[i] > 0 && x[i] <1 
            out[i] = 1
        end
    end
    out
end

####Monte Carlo simulations of Eq. 4

function utility_simulations_Gaussian(N,t_vec,σ,σ_0,μ_0=0.5,N_iter=1E6, default = false)
    #t_vec already complies with the constraints and its length should be as large as N
    utility = 0
    σp_t = zeros(N)
    for j in 1:N
        σp_t[j] = σ_0*σ/sqrt(t_vec[j]*σ_0^2 + σ^2)
    end
    #Simulate N_iter times
    for i in 1:N_iter
        mu_best = -999999 #μ_0
        mu_ut = 0
        #Simulate the N races
        for j in 1:N
            mu = rand(Normal(μ_0,σ_0))
            x = mu*t_vec[j]+σ*sqrt(t_vec[j])*rand(Normal(0,1))
            #Posterior expected value of drift
            mu_post = (σp_t[j]^2)*(x/(σ^2) + μ_0/(σ_0^2))
            if mu_post > mu_best
                mu_best = mu_post
                mu_ut = mu
            end
        end
        #If we let a default option, we compare the best sampled with the prior mean,
        #if prior mean wins, we choose an unsampled option
        if default == true
            if mu_best < μ_0 
                mu = rand(Normal(μ_0,σ_0))
                mu_ut = mu
            end
        end
        utility += mu_ut
    end
    utility/N_iter
end

function utility_simulations_flat(N,t_vec,σ,N_iter=1E6,default = false)
    #t_vec already complies with the constraints and its length should be as large as N
    utility = 0
    mu_0 = 0.5
    for i in 1:N_iter
        mu_best = 0
        x_best = -999999
        for j in 1:N
            mu = rand()
            x = mu*t_vec[j]+σ*sqrt(t_vec[j])*rand(Normal(0,1))
            #Note that we maximize over evidence here, 
            #so it only works for even allocation
            if x > x_best
                mu_best = mu
                x_best = x
            end
        end
        #If we let a default option, we compare the best sampled with the prior mean,
        #if prior mean wins, we choose an unsampled option
        if default == true
            if mu_best < mu_0 
                mu_best = rand()
            end
        end
        utility += mu_best/N_iter
    end
    utility
end

function utility_simulations_bimodal_Gaussian(M,t_vec,σ,σ_0,μ_1,μ_2,N_iter=1E6,default = false)
    #t_vec already complies with the constraints and its length should be as large as N
    utility = 0
    var = 0
    mu_0 = (μ_1 + μ_2)/2
    for i in 1:N_iter
        mu_best = 0
        x_best = -999999
        #Simulate the N races
        for j in 1:M
            if rand() <= 1/2
                mu = rand(Normal(μ_1,σ_0))
            else
                mu = rand(Normal(μ_2,σ_0))
            end
            x = mu*t_vec[j]+σ*sqrt(t_vec[j])*rand(Normal(0,1))
            #Note that we maximize over evidence here, 
            #so it only works for even allocation. 
            #Easy to extend for uneven, using expected_mu
            if x > x_best
                mu_best = mu
                x_best = x
            end
        end
        #If we let a default option, we compare the best sampled with the prior mean,
        #if prior mean wins, we choose an unsampled option
        if default == true
            if mu_best < mu_0 
                if rand() <= 1/2
                    mu = rand(Normal(μ_1,σ_0))
                else
                    mu = rand(Normal(μ_2,σ_0))
                end
                mu_best = mu
            end
        end
        utility += mu_best/N_iter
        #Prior code to also compute variance 
#         delta = mu_best - utility
#         utility += delta/i
#         delta2 = mu_best - utility
#         var += delta*delta2
#         #utility += mu_best/N_iter
    end
    utility #, sqrt(var/N_iter)
end

###### Numerical integration of analytical expressions 

function utility_analytical_Gaussian(C,N,Δu = 0.001,u_max = 20)
    u = collect(-u_max:Δu:u_max)
    total_u = length(u)
    out = 0
    ##Numerical integration, eq. 10
    for i in 1:total_u
        out += N*Δu*u[i]*exp(-u[i]^2/2)*((1+erf(u[i]/sqrt(2)))/2)^(N-1)
    end
    out *= 1/sqrt((1+N/C)*2*pi)
    out
end

#This function is not used in paper
function utility_arbitrary_N2(μ_0,σ_0,σ,t_vec)
    σ_t = zeros(2)
    for i in 1:2
        σ_t[i] = σ_0*sqrt(t_vec[i])/sqrt(σ_0^2*t_vec[i] + σ^2)
    end
    μ_0 + (σ_0/sqrt(2*pi))*sqrt(σ_t[1]^2+σ_t[2]^2)
end

"""
    expected_mu(x,t,σ)

Computes the posterior expectation of a drift, given evidence `x` and time `t`
for a uniform prior between 0 and 1.
"""
function expected_mu(x,t,σ)
    num = exp(-x^2/(2*σ^2*t))-exp(-(t*(1-x/t)^2)/(2*σ^2))
    den = (erf(sqrt(t)*(1-x/t)/(σ*sqrt(2)))+erf(x/(σ*sqrt(2*t))))/2
    x/t+(σ/sqrt(2*pi*t))*num/den
end

prob_density(x,t,σ)=(erf(sqrt(t)*(1-x/t)/(σ*sqrt(2)))+erf(x/(σ*sqrt(2*t))))/(2*t)

cum_flat(x,t,σ) = (1+(x/t)*(erf(x/(sqrt(2*σ^2*t)))-erf((x-t)/(sqrt(2*σ^2*t))))+erf((x-t)/(sqrt(2*σ^2*t)))+sqrt(2*σ^2/(t*pi))*(exp(-x^2/(2*σ^2*t))-exp(-(x-t)^2/(2*σ^2*t))))/2


"""
    utility_analytical_flat(C,M,σ,N_points = 1E5)

Numerical integration of the expected utility Eq. (20) for a uniform prior, 
given capacity `C`, `M` sampled accumulators and noise `σ`. 
Uses N_points = 1E5 points by default.
"""
function utility_analytical_flat(C,M,σ,N_points = 1E5)
    σ_0 = 1/sqrt(12)
    t = C*σ^2/(σ_0^2*M)
    C /= σ_0^2
    x_max = 1000
    Δx = 2*x_max/N_points
    x = collect(-x_max:Δx:x_max)
    total_x = length(x)
    out = 0
    cumulative = 0
    for i in 1:total_x
        cumulative = cum_flat(x[i]*sqrt(2*σ^2*t),t,σ)
        #Integrand in paper
        out += cumulative^(M-1)*(cumulative - (1/2 + erf(x[i]-sqrt(C/(2*M)))/2))*Δx
    end
    out *= M*sqrt(2*M/C)
    out
end

"""
    utility_analytical_bimodal(C,M,σ,σ_0,μ_1,μ_2,N_points)

Numerical integration of the expected utility Eq. (21) for a bimodal prior, 
given capacity ``C``, ``M`` sampled accumulators, noise ``σ``, common variance ``σ_0``,
and means ``μ_1,μ_2`` for each mode.
Uses `N_points = 1E5` points by default.
"""
function utility_analytical_bimodal(C,M,σ,σ_0,μ_1,μ_2,N_points = 1E5)
    μ_0 = (μ_1 + μ_2)/2
    σ_B = sqrt(σ_0^2 + (μ_1^2 + μ_2^2)/2 - μ_0^2)
    t = C*σ^2/(M*σ_B^2)
    σ_t2 = σ_0^2*σ^2/(t*σ_0^2 + σ^2)
    σ_G2 = σ^2*t + σ_0^2*t^2
    σ_x = sqrt(σ_G2 + (μ_1^2 + μ_2^2)*t^2/2 - μ_0^2 * t^2)
    #Integrates up to 1000 times standard deviation of x
    δx = 1000*σ_x/N_points
    x_vec = collect(μ_0*t - 500*σ_x:δx:μ_0*t + 500*σ_x) 
    out = 0
    ##Numerical integration
    for (idx,x) in enumerate(x_vec)
        exp_μ1 = σ_t2*x/σ^2 + σ_t2*μ_1/σ_0^2
        exp_μ2 = σ_t2*x/σ^2 + σ_t2*μ_2/σ_0^2
        exp_μ = exp_μ1*normal(x,μ_1*t,σ_G2)/2 + exp_μ2*normal(x,μ_2*t,σ_G2)/2
        out += M*(δx)*(1/2 +(erf((x-μ_1*t)/sqrt(2*σ_G2))+erf((x-μ_2*t)/sqrt(2*σ_G2)))/4)^(M-1)*exp_μ
    end
    out
end

#########################
"""
    compute_even_allocation(σ,μ_0,σ_0_G,N_points,utility_plots = false,N_sim_scale = 1E6, sim = false)

Computes expected utility for Gaussian, uniform and bimodal priors.
It has two general methods: `utility_plots = true` refers to figure 3 and `false` is for figure 4.
# Arguments
- `σ_0_G`: the standard deviation of Gaussian and common stdev for bimodal.
- `μ_0`: the mean for Gaussian.
- `N_sim_scale`: a measure of the number of Monte Carlo simulations to be used for each prior.
- `N_points`: the number of points used for numerical integration of analytical expressions.
- `sim`: a boolean to compute simulations or not.

# Output
If `utility_plots = true`, it returns a tuple `(C,M, Us_simG,Us_anG)`.
- `C` is the several capacities at which we compute expected utility.
- `M` is the number of sampled accumulators. 
- `Us_simG` is the simulated expected utility.
- `Us_anG` is the numerically integrated expected utility.
For Gaussian prior.

If `utility_plots = false`, it returns a tuple containing the optimal Ms,
`(C,M_optG, M_optF, M_optB, M_opt_simG, M_opt_simF, M_opt_simB, M_opt_simG_def, M_opt_simF_def, M_opt_simB_def)`
- `G,F,B` refer to Gaussian, uniform and bimodal priors, respectively.
- `def` subscript refers to the case when a default option can be chosen.
- `sim` referes to simulations, its absence means it is numerical integration.
"""
function compute_even_allocation(σ,μ_0,σ_0_G,N_points,utility_plots = false,N_sim_scale = 1E6, sim = false)
    σ_0_F = 1/sqrt(12)
    μ_1 = -1#-4.5
    μ_2 = 2#5.5
    σ_B = sqrt(σ_0_G^2 + (μ_1^2 + μ_2^2)/2 - mean([μ_1,μ_2])^2)
    #Integration interval for Gaussian
    u_max = 100
    #Integration step
    Δu = u_max/N_points
    if utility_plots == true
        powers = collect(0:4)
        bases = [1,2,3,5]
        #Capacities to test
        C = [1E0,1E1,1E2] 
    else
        powers = collect(0:2)
        bases = [1,2,3,4,5,6,7,8,9]
        #Capacities to test
        #C = [1E-4,3E-4,1E-3,3E-3,1E-2,3E-2,1E-1,3E-1,1,3,1E1,3E1,1E2,3E2,1E3,3E3,1E4,3E4,1E5,3E5,1E6,3E6,1E7,3E7,1E8]
        C = [1E-4,1E-3,1E-2,1E-1,1,1E1,1E2,1E3,1E4,1E5,1E6,1E7]
    end
    M = zeros(length(powers)*length(bases))
    for p in 1:length(powers)
        for b in 1:length(bases)
        M[length(bases)*(p-1)+b] = bases[b]*10^powers[p]
        end
    end
    #Calculate T for all priors
    T_G = C.*((σ^2)/(σ_0_G^2))
    T_F = C.*((σ^2)/(σ_0_F^2))
    T_B = C.*((σ^2)/(σ_B^2))
    M_optF = zeros(length(C))
    M_optB = zeros(length(C))
    M_opt_simF = zeros(length(C))
    M_opt_simB = zeros(length(C))
    M_opt_simF_def = zeros(length(C))
    M_opt_simB_def = zeros(length(C))
    M_optG = zeros(length(C))
    M_opt_simG = zeros(length(C))
    M_opt_simG_def = zeros(length(C))
    Us_simG = zeros(length(C),length(M))
    Us_anG_plot = zeros(length(C),length(M))
    for i in 1:length(C)
        if utility_plots == false
            #The Ms that will be used to search for the maximum
            M = [1,2,3,4,5,6,7,8,9]
            #Depending on value of capacity, add more Ms
            if C[i] > 1 && C[i] < 1000
                powers = collect(1:Int64(round(log10(C[i])))+1)
                if sim == true
                    bases = collect(1:0.5:9)
                else
                    bases = collect(1:0.5:9)
                end
                for p in powers 
                    for base in bases
                        push!(M,Int64(round(base*10^p)))
                    end
                end
            elseif C[i] >= 1000
                #After 1E2 we see the sublinearity so we only search up to C
                powers = collect(1:Int64(round(log10(C[i]))))
                if sim == true
                    bases = collect(1:0.5:9)
                else
                    bases = collect(1:0.5:9)
                end
                for p in powers 
                    for base in bases
                        push!(M,Int64(round(base*10^p)))
                    end
                end
                #N_sim = 1E1
            end
        end
        Us_anG = zeros(length(M))
        Us_anB = zeros(length(M))
        Us_anF = zeros(length(M))
        Us_simG_2 = zeros(length(M))
        Us_simF = zeros(length(M))
        Us_simB = zeros(length(M))
        Us_simF_def = zeros(length(M))
        Us_simB_def = zeros(length(M))
        Us_simG_def = zeros(length(M))
        #Number of simulations depends on value of capacity.
        #Smaller capacities need more simulations.
        N_sim = Int64(round(N_sim_scale ./ sqrt(C[i])))
        for n in 1:length(M)
            #Analytical
            Us_anG[n] = utility_analytical_Gaussian(C[i],M[n],Δu,u_max)*σ_0_G + μ_0
            Us_anF[n] = utility_analytical_flat(C[i],M[n],σ,N_points)
            Us_anB[n] = utility_analytical_bimodal(C[i],M[n],σ,σ_0_G,μ_1,μ_2,N_points)
            if utility_plots == true
                Us_anG_plot[i,n] = Us_anG[n]
                t_vec = ones(Int64(M[n]))*T_G[i]/M[n]
                Us_simG[i,n] = utility_simulations_Gaussian(Int64(M[n]),t_vec,σ,σ_0_G,μ_0,N_sim_scale)
            else
                #Simulations
                if sim == true 
                    if Int64(round(log10(C[i])))%2 == 0
                        t_vec = ones(Int64(M[n]))*T_G[i]/M[n]
                        Us_simG_2[n] = utility_simulations_Gaussian(Int64(M[n]),t_vec,σ,σ_0_G,μ_0,N_sim)
                        Us_simF[n] = utility_simulations_flat(Int64(M[n]),t_vec,σ,N_sim)
                        t_vec_B = ones(Int64(M[n]))*T_B[i]/M[n]
                        Us_simB[n] = utility_simulations_bimodal_Gaussian(Int64(M[n]),t_vec_B,σ,σ_0_G,μ_1,μ_2,N_sim)
                    else
                        t_vec = ones(Int64(M[n]))*T_G[i]/M[n]
                        Us_simG_def[n] = utility_simulations_Gaussian(Int64(M[n]),t_vec,σ,σ_0_G,μ_0,N_sim,true)
                        Us_simF_def[n] = utility_simulations_flat(Int64(M[n]),t_vec,σ,N_sim,true)
                        t_vec_B = ones(Int64(M[n]))*T_B[i]/M[n]
                        Us_simB_def[n] = utility_simulations_bimodal_Gaussian(Int64(M[n]),t_vec_B,σ,σ_0_G,μ_1,μ_2,N_sim,true)
                    end
                end
            end
        end
        M_optG[i] = M[findmax(Us_anG[:])[2]]
        M_optF[i] = M[findmax(Us_anF[:])[2]]
        M_optB[i] = M[findmax(Us_anB[:])[2]]
        M_opt_simG[i] = M[findmax(Us_simG_2[:])[2]]
        M_opt_simF[i] = M[findmax(Us_simF[:])[2]]
        M_opt_simB[i] = M[findmax(Us_simB[:])[2]]
        M_opt_simG_def[i] = M[findmax(Us_simG_def[:])[2]]
        M_opt_simF_def[i] = M[findmax(Us_simF_def[:])[2]]
        M_opt_simB_def[i] = M[findmax(Us_simB_def[:])[2]]
    end
    #Output for figure 3
    if utility_plots == true 
        out = M, Us_simG,Us_anG_plot
    #Output for figure 4
    else
        out = M_optG, M_optF, M_optB, M_opt_simG, M_opt_simF, M_opt_simB, M_opt_simG_def, M_opt_simF_def, M_opt_simB_def
    end
    C,out
end

"""
    plot_utility_even(C,M,Us_sim,Us_an)

Plots Fig. 3. Input is output from `compute_even_allocation` with `utility_plots = true`.
"""
function plot_utility_even(C,M,Us_sim,Us_an)
    widthCM = 12
    heightCM = 8
    f = figure(figsize=(widthCM/2.54, heightCM/2.54), dpi=72)
    ax = gca()
    #ax.set_frame_on(false)
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    # indices = [1,2,3]
    indices = [6,8,10]
    colors = get_cmap("gray_r")
    cindex = [0.2,0.5,0.9]
    labels = ["C = 1","C = 10","C = 100"]
    ax.semilogx(M,0.5.*ones(length(M)),"--",linewidth=2,color = "gray")
    for (idx,c) in enumerate(C)
        ax.semilogx(M,Us_an[idx,:],"-", color = colors(cindex[idx]),label = labels[idx])
        ax.semilogx(M,Us_sim[idx,:],"o",markersize=4, color = colors(cindex[idx]))
        idx_max = findmax(Us_an[idx,:])[2]
        ax.semilogx(M[idx_max],Us_an[idx,idx_max],"o",color = "royalblue",markersize = 6)
    end

    tight_layout(rect = [0.08, 0.08, 1, 1])

    ax.set_xlim([0,1E5])
    ax.set_ylim([0.0,2.7])
    ax.set_xticks([1E0,1E1,1E2,1E3,1E4,1E5])
    ax.set_yticks([0.0,0.5,1.0,1.5,2.0,2.5])
    ax.tick_params(labelsize=16,direction = "out",top=false)

    ax.set_xlabel(L"Number $M$ of sampled accumulators",fontsize = 16)
    ax.set_ylabel("Expected utility",rotation=90,fontsize = 16)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[end:-1:1],labels[end:-1:1],loc=1,fontsize=14, ncol = 1, frameon =false)
end

"""
    plot_Mopt_vs_C(C,M_optG,M_optF,M_optB,M_opt_simG,M_opt_simF,M_opt_simB, M_opt_simG_def, M_opt_simF_def, M_opt_simB_def, sim = false)

Plots Fig. 4. Input is output from `compute_even_allocation` with `utility_plots = false`.
"""
function plot_Mopt_vs_C(C,M_optG,M_optF,M_optB,M_opt_simG,M_opt_simF,M_opt_simB, M_opt_simG_def, M_opt_simF_def, M_opt_simB_def, sim = false)
    start1 = 1
    oend = 6#11
    start2 = oend + 1
    eend = length(C)
    C_max = length(C)
    #Fit line for uniform prior case
    b2,m2=linear_fit(log.(C[start2:eend]),log.(M_optF[start2:eend]))
    M_fitted2 = exp.(log.(C[start2:eend]).*m2 .+ b2)

    widthCM = 16
    heightCM = 10
    f = figure(figsize=(widthCM/2.54, heightCM/2.54), dpi=72)
    ax = gca()
    rc("text", usetex=true)
    rc("font",family="serif", serif = "Computer Modern")
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    paso_plot = 1
    
    #Plot the inset
    axins = ax.inset_axes([0.13,0.55,0.45,0.35] )
    x = collect(-1.8:0.08:2.8)
    axins.plot(x[7:1:end-7],plot_uniform(x[7:1:end-7]),"s-",markersize = 2,linewidth = 1, color = "violet")
    axins.plot(x,normal.(x,0.5,1/12),"o-",markersize = 2,linewidth = 1, color = "royalblue")
    axins.plot(x[1:2:end],0.5.*normal.(x[1:2:end],-1,1/12) .+ 0.5.*normal.(x[1:2:end],2,1/12),"v-",markersize = 3,linewidth = 1, color = "limegreen")
    axins.set_ylim([0,1.5])
    axins.set_xticks([-2,-1,0,1,2,3])
    axins.set_yticks([0,1])
    axins.set_ylabel("prior pdf",fontsize = 15)
    axins.set_xlabel(L"\mu",fontsize = 16)
    axins.tick_params(labelsize=14,direction = "out",top=false)
    
    C_5 = [3E-5,1E1,1E7]
    ax.plot(C_5,5*ones(length(C_5)),"k--",linewidth = 1.5, label=L"M = 5")
    
    #Numerical integration
    ax.plot(C[start1:paso_plot:eend],M_optF[start1:paso_plot:eend],"-", markersize = 5,linewidth =2, color = "violet")
    ax.plot(C[start1:paso_plot:eend],M_optG[start1:paso_plot:eend],"-", markersize = 5,linewidth =2, color = "royalblue")
    ax.plot(C[start1:paso_plot:eend],M_optB[start1:paso_plot:eend],"-", markersize = 5,linewidth =2, color = "limegreen")

    #Asymptotic expressions
    ax.plot(C[5:eend],(C[5:eend])./(lambertw.(C[5:eend])),label=L"$C/W(C)$","--",linewidth = 2, color = "orangered")
    ax.plot(C[start2:eend],M_fitted2,label=string("exp = ",round(m2,digits=2)),"--",linewidth = 2, color = "grey")
    
    if sim == true
        ax.plot(C[1:2:C_max],M_opt_simF[1:2:C_max],"s", markersize = 5, color = "violet")
        ax.plot(C[1:2:C_max],M_opt_simG[1:2:C_max],"o", markersize = 5, color = "royalblue")
        ax.plot(C[1:2:C_max],M_opt_simB[1:2:C_max],"v", markersize = 5, color = "limegreen")

        ax.plot(C[2:2:C_max],M_opt_simF_def[2:2:C_max],"D", markersize = 6, color = "violet")
        ax.plot(C[2:2:C_max],M_opt_simG_def[2:2:C_max],"D", markersize = 5, color = "royalblue")
        ax.plot(C[2:2:C_max],M_opt_simB_def[2:2:C_max],"D", markersize = 4, color = "limegreen")
        
    end
    ax.set_xscale("log")
    ax.set_yscale("log")

    tight_layout(rect = [0.06, 0.08, 1, 0.9])

    ax.set_xlabel(L"Capacity $C$",fontsize = 18) 
    ax.set_ylabel(L"Optimal $M$",rotation=90,fontsize = 18)

    xticks([1E-4,1E-2,1E0,1E2,1E4,1E6])
    ax.set_yticks([1E0,1E2,1E4,1E6,1E8])
    ax.set_ylim([1,1E8])
    ax.tick_params(labelsize=16,direction = "out",top=false)

    ax.legend(loc=1,fontsize=14,ncol = 3, bbox_to_anchor=(0.36, 1.1,0.6,0.05),frameon =false)
end


"""
    utility_directions(T,σ,μ_0,σ_0,N_ts,N_sim,prior = "Gaussian", default = false)

Computes expected utility in crystallographic directions from Fig. 5.
It has the option to calculate for the case when a default option can be chosen.
For now it should only be used for the Gaussian prior.
`N_ts` is the number of points between symmetrical critical points.
"""
function utility_directions(T,σ,μ_0,σ_0,N_ts,N_sim,prior = "Gaussian", default = false)
    U = zeros(N_ts * 7)
    if prior == "Gaussian"
            for t in 1:N_ts
                t_vec = [T - T/2 *t/N_ts,T/2 * t/N_ts]
                U[t] = utility_simulations_Gaussian(2,t_vec,σ,σ_0,μ_0,N_sim,default)
                t_vec1 = [T/2 - T/6 *t/N_ts,T/2 - T/6 *t/N_ts,T/3 * t/N_ts]
                U[t + N_ts] = utility_simulations_Gaussian(3,t_vec1,σ,σ_0,μ_0,N_sim,default)
                t_vec2 = [T/3 - T/12 *t/N_ts,T/3 - T/12 *t/N_ts,T/3 - T/12 *t/N_ts,T/4 * t/N_ts]
                U[t + 2*N_ts] = utility_simulations_Gaussian(4,t_vec2,σ,σ_0,μ_0,N_sim,default)
                t_vec3 = [T/4 - T/20 *t/N_ts,T/4 - T/20 *t/N_ts,T/4 - T/20 *t/N_ts,T/4 - T/20 *t/N_ts,T/5 * t/N_ts]
                U[t + 3*N_ts] = utility_simulations_Gaussian(5,t_vec3,σ,σ_0,μ_0,N_sim,default)
                t_vec4 = [T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/6 * t/N_ts]
                U[t + 4*N_ts] = utility_simulations_Gaussian(6,t_vec4,σ,σ_0,μ_0,N_sim,default)
                t_vec5 = [T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/7 * t/N_ts]
                U[t + 5*N_ts] = utility_simulations_Gaussian(7,t_vec5,σ,σ_0,μ_0,N_sim,default)
                t_vec6 = [T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/8 * t/N_ts]
                U[t + 6*N_ts] = utility_simulations_Gaussian(8,t_vec6,σ,σ_0,μ_0,N_sim,default)
            end
    elseif prior == "uniform"
        for t in 1:N_ts
            t_vec = [T - T/2 *t/N_ts,T/2 * t/N_ts]
            U[t] = utility_simulations_flat(2,t_vec,σ,N_sim,default)
            t_vec1 = [T/2 - T/6 *t/N_ts,T/2 - T/6 *t/N_ts,T/3 * t/N_ts]
            U[t + N_ts] = utility_simulations_flat(3,t_vec1,σ,N_sim,default)
            t_vec2 = [T/3 - T/12 *t/N_ts,T/3 - T/12 *t/N_ts,T/3 - T/12 *t/N_ts,T/4 * t/N_ts]
            U[t + 2*N_ts] = utility_simulations_flat(4,t_vec2,σ,N_sim,default)
            t_vec3 = [T/4 - T/20 *t/N_ts,T/4 - T/20 *t/N_ts,T/4 - T/20 *t/N_ts,T/4 - T/20 *t/N_ts,T/5 * t/N_ts]
            U[t + 3*N_ts] = utility_simulations_flat(5,t_vec3,σ,N_sim,default)
            t_vec4 = [T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/5 - T/30 *t/N_ts,T/6 * t/N_ts]
            U[t + 4*N_ts] = utility_simulations_flat(6,t_vec4,σ,N_sim,default)   
            t_vec5 = [T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/6 - T/42 *t/N_ts,T/7 * t/N_ts]
            U[t + 5*N_ts] = utility_simulations_flat(7,t_vec5,σ,N_sim,default) 
            t_vec6 = [T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/7 - T/56 *t/N_ts,T/8 * t/N_ts]
            U[t + 6*N_ts] = utility_simulations_flat(8,t_vec6,σ,N_sim,default) 
        end
    end
    U
end


"""
    plot_directions(N_ts,U_directions)

Plots panel D in figure 4. Input is the output from `utility_directions`.
"""
function plot_directions(N_ts,U_directions)
    theticks = [0,N_ts,2*N_ts,3*N_ts,4*N_ts,5*N_ts,6*N_ts,7*N_ts]
    ticklabels = [L".",L".",L".",L"\mathbf{t}_4^e",L"\mathbf{t}_5^e",L"\mathbf{t}_6^e",L"\mathbf{t}_7^e",L"\mathbf{t}_8^e"]
    widthCM = 10
    heightCM = 6
    f = figure(figsize=(widthCM/2.54, heightCM/2.54), dpi=72)
    ax = gca()
    tight_layout(rect = [0.1, 0.1, 1, 1])
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    plot(collect(1:7*N_ts),U_directions,"-o",color="darkorange",markersize = 3)
    plot(4*N_ts,U_directions[4*N_ts],"o",markersize = 6, color = "red")
    ax.set_xticks(theticks)
    ax.grid(true,"major","x",linestyle="--")
    ax.set_xticklabels(ticklabels)
    #ax.set_yticks([0.62,0.64,0.66])
    ax.tick_params(labelsize=16,direction = "out",top=false)
    ax.set_xlabel("Simplex directions",fontsize = 16)
    ax.set_ylabel("Expected utility",rotation=90,fontsize = 16)
end

"""
    plot_utility_triangle(T,σ,μ_0,σ_0,N_xy,N_z,N_sim = 1E6,colorbar = false)

Computes and plots the expected utility in the triangle in Fig. 4c, as well as colorplot.
"""
function plot_utility_triangle(T,σ,μ_0,σ_0,N_xy,N_z,N_sim = 1E6,colorbar = false)
    # First compute utility in 3D, random points
    U_3D = zeros(N_xy,N_z)
    ts = []
    x = zeros(N_xy, N_z)
    y = zeros(N_xy, N_z)
    z = zeros(N_xy,N_z)
    for k in 1:N_z
        zeta = T*(k-1)/(N_z-1)
        border = T - zeta
        #print("border = ", border, "\n")
        for j in 1:N_xy
            z[j,k] = zeta
            x[j,k] = border*(j-1)/(N_xy-1)
            #print("x = ", x[j], "\n")
            y[j,k] = border - x[j,k]
            if y[j,k] < 0
                y[j,k] = 0
            end
            if x[j,k] < 0
                x[j,k] = 0
            end
            if z[j,k] < 0
                z[j,k] = 0
            end
            U_3D[j,k] = utility_simulations_Gaussian(3,[x[j,k],y[j,k],z[j,k]],σ,σ_0,μ_0,N_sim)
        end
    end
    #Construct colors from values of utility
    colorU = (U_3D .- mean(U_3D))./(maximum(U_3D)- mean(U_3D));
    colorss = zeros(N_xy,N_z,3)
    for k in 1:N_z
        for i in 1:N_xy
            colorss[i,k,1] = exp(-(colorU[i,k] - mean(U_3D))^2)#1 - abs(colorU[i,k] - 1.0)
            colorss[i,k,2] = exp(-(colorU[i,k] + 0.3)^2)#1 - abs(colorU[i,k] - 0.3)
            colorss[i,k,3] = exp(-(colorU[i,k] + mean(U_3D))^2)#1 - abs(colorU[i,k] - 0.0)
        end
    end

    if colorbar == true
        #Build colorbar from values
        Ncolors = 50
        colorbarx = zeros(Ncolors,Ncolors)
        colorsB = zeros(Ncolors,Ncolors,3)
        colorbary = zeros(Ncolors,Ncolors) 
        for k in 1:Ncolors
            for i in 1:Ncolors
                colorbary[i,k] = ((k-1)/(Ncolors-1) - mean(U_3D))/(1 - mean(U_3D))
                colorbarx[i,k] = ((i-1)/(Ncolors-1) - mean(U_3D))/(1 - mean(U_3D))
                colorsB[i,k,1] = exp(-abs(colorbarx[i] - mean(U_3D))^2)#1 - abs(colorbarx[i] - 1.0)
                colorsB[i,k,2] = exp(-abs(colorbarx[i] + 0.3)^2)#1 - abs(colorbarx[i] - 0.3)
                colorsB[i,k,3] = exp(-abs(colorbarx[i] + mean(U_3D))^2)#1 - abs(colorbarx[i] - 0.0)
            end
        end
        surf(colorbarx,colorbary,zeros(Ncolors,Ncolors), facecolors = colorsB)
        gca().view_init(-90,0)
        xticks([])
        yticks([])
        zticks([])
        axis("off");
    else
        #Plot surface
        surf(x,y,z, facecolors = colorss, alpha = 0.9)
        xticks([0.0,0.05,0.1])
        gca().view_init(20,15)
        gca().tick_params(labelsize=12,direction = "out",top=false)
    end
end

#### Projected gradient ascent method

function random_initialization(M_init, threshold = 1E-5)
    t_vec = zeros(M_init)
    remaining = 1
    counter = 0
    for i in 1:M_init-1
        #If what is left is less than our threshold for deactivating dimensions, allocate the rest evenly
        if remaining/(M_init - counter) > threshold
            #A component of the initial cannot be less than threshold, in order to activate all dimensions
            t_vec[i] = (rand()*(remaining-threshold) + threshold) 
            remaining = remaining - t_vec[i]
            counter += 1
        else
            t_vec[i] = remaining/(M_init - counter)
        end
    end
    t_vec[end] = 1 - sum(t_vec)
    #Return sorted allocation vector
    sort(t_vec,rev=true)
end

"""
    deriv_Gaussian(i,t_vec,σ,σ_0,delta,N_points = 5001)

Function that numerically integrates the derivative of utility along direction `i`.
"""
function deriv_Gaussian(i,t_vec,σ,σ_0,delta,N_points = 5001)
    idx = []
    σs = zeros(length(t_vec))
    for j in 1:length(t_vec)
        #Only take directions where the allocated time is positive
        if t_vec[j] > delta
            σs[j] = sqrt(σ_0^2*t_vec[j]/(σ_0^2*t_vec[j] + σ^2))
            push!(idx,j)
        end
    end
    σ_max = σs[i]
    z_max = 1000*σ_max
    dz = 2*z_max/N_points
    z = collect(-z_max:dz:z_max)
    total_z = length(z)
    out = 0
    dσdt = σ_0*σ^2/(2*sqrt(t_vec[i])*(σ_0^2*t_vec[i] + σ^2)^(3/2))
    ##Numerical integration
    for l in 1:total_z
        prod1 = 1
        sum = 0
        for j in idx
            prod2 = 1
            if j != i
                prod1 *= (1 + erf(z[l]/(sqrt(2)*σs[j])))/2
            end
            for k in idx
                if k != j && k != i
                    prod2 *= (1 + erf(z[l]/(sqrt(2)*σs[k])))/2
                end
            end
            sum += normal(z[l],0,σs[j]^2)*prod2
        end
        out += dz * (z[l]/σs[i]) * normal(z[l],0,σs[i]^2) * dσdt * ((z[l]^2/(σs[i]^2) - 1)*prod1 - z[l] * sum)
    end
    out *= σ_0
    out
end

"""
    ascent(t_vec,T,σ,σ_0,η = 0.1,sign_ascent = 1,N_iter = 1000, verbose = false)

Function that performs the whole iterative algorithm.
Output: the final allocation, mean of allocation at each step, std of allocation at each step.
"""
function ascent(t_vec,T,σ,σ_0,η = 0.1,sign_ascent = 1,N_iter = 1000, verbose = false)
    M = length(t_vec)
    P = Matrix(I,M,M) - ones(M,M)./M   
    delta = 1E-5*T
    ids = collect(1:M)
    t_vec_mean = zeros(N_iter)
    t_vec_std = zeros(N_iter)
    for k in 1:N_iter
        grad = zeros(length(t_vec))
        deltat = zeros(length(t_vec))
        idx = copy(ids)
        eps = 0.1
        if rand() < eps
            l = rand(idx)
            m = rand(idx)
            #size of the perturbation
            t_vec[l] -= T/1000
            t_vec[m] += T/1000
        end
        ids = []
        for (id,val) in enumerate(idx)
            #Here we add dimensions to the active set, and only calculate the gradient for the inactive set
            if t_vec[val] > delta
                grad[val] = deriv_Gaussian(val,t_vec,σ,σ_0,delta)
                push!(ids,id)
            end
        end
        #Sort so that active set dimensions are left at the end of t_vec, in order to compare with t_vec_compare
        p = sortperm(t_vec,rev=true)
        t_vec = t_vec[p]
        M = length(ids)
        P = Matrix(I,M,M) - ones(M,M)./M   
        deltat[p[ids]] = P*grad[p[ids]]
        #In this section we see if the step size is appropriate
        not_feasible = true
        η_p = η
        t_vec_pot = zeros(M)
        while not_feasible == true
            t_vec_pot = t_vec[1:M] + sign_ascent*η_p*deltat[p[ids]]
            #If there is any component that would give negative value 
            #and step size is still big enough, then decrease the step size.
            if all(t_vec_pot .> 0) == true || η_p < 1E-12
                not_feasible = false
            else
                η_p  = 0.1*η_p
            end
        end
        #We then have a new allocation vector
        t_vec[1:M] = t_vec_pot 
        for i in M+1:length(t_vec)
            #Now, if there is any component that is basically zero, 
            #then make it exactly zero and give its value to a random component.
            if t_vec[i] < delta
                t_vec[rand(1:M)] += t_vec[i]
                t_vec[i] = 0
            end
        end
        sort!(t_vec,rev=true)
        #Verbose gives us at each step the values of the allocation vector
        if verbose == true
            println("k = ", k)
            println("t_vec = ", t_vec./T,", sum = ", sum(t_vec./T))
        end
        t_vec_mean[k] = mean(t_vec[1:M])
        t_vec_std[k] = std(t_vec[1:M])
    end
    t_vec[ids],t_vec_mean,t_vec_std
end

"""
    compute_ascent_CV(Ts,σ,σ_0,N_points,initial_M,sign_ascent = 1, η = T/100,N_steps = 2000)

Computes the `ascent` function for several `T` and `N_points` random initial conditions.
`initial_M` is the vector that gives the initial number of active dimensions for each `T`.
"""
function compute_ascent_CV(Ts,σ,σ_0,N_points,initial_M,sign_ascent = 1, η = T/100,N_steps = 2000)
    meanss = zeros(length(Ts),N_points,N_steps)
    stdss = zeros(length(Ts),N_points,N_steps)
    for (j,T) in enumerate(Ts)
        for i in 1:N_points
            #Initialization for each condition is random
            #and the number of active dimensions is given by initial_M[j]
            t_vec = T.*random_initialization(initial_M[j],1E-4*T)
            t_star, means, stds = ascent(t_vec,T,σ,σ_0,η*T,sign_ascent,N_steps)
            meanss[j,i,:] = means
            stdss[j,i,:] = stds
        end
    end
    meanss,stdss
end

"""
    plot_ascent_random(mean,std,Ts,N_points)

Plots Fig. 5e. Input is the output of `compute_ascent_CV`.
"""
function plot_ascent_random(mean,std,Ts,N_points)
    widthCM = 10
    heightCM = 6
    f = figure(figsize=(widthCM/2.54, heightCM/2.54), dpi=72)
    ax = gca()
    tight_layout(rect = [0.1, 0.1, 1, 1])
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    colors = ["navy","red","green"]
    alphas = [0.5,0.6,0.7]
    for (idx,T) in enumerate(Ts)
        counter = 0
        for i in 1:N_points-1
            if i == 1
                ax.plot(std[idx,i,:]./mean[idx,i,:], label = "C = $(T)", color = colors[idx], alpha = alphas[idx])
            else
                ax.plot(std[idx,i,:]./mean[idx,i,:], color = colors[idx], alpha = alphas[idx])
            end
        end
    end
    ax.legend(loc = 1, fontsize = 14, frameon = false)
    ax.tick_params(labelsize=16,direction = "out",top=false)
    ax.set_xlabel(L"Step $k$",fontsize = 16)
    ax.set_ylabel(L"CV$(\mathbf{t}^{(k)})$",rotation=90,fontsize = 16)
end



###### Other functions for the ascent algorithm that are not used in paper.
function compute_ascent_distances_symmetric(T,σ,σ_0,Ms,t_vec_compare,initial_M,sign_ascent = 1, η = T/100,N_steps = 2000,N_reps = 10)
    #Embed all points in M+1 space
    dists = zeros(length(Ms),N_steps)
    vars = zeros(length(Ms),N_steps)
    delta_border = T/10000
    for i in 1:length(Ms)
        for j in 1:N_reps
            #Initialization for each condition is at symmetric point 
            t_vec = zeros(initial_M) .+ delta_border
            #However, we need to activate the rest of dimensions
            t_vec[1:Ms[i]] .= T/Ms[i] - (initial_M-Ms[i])*delta_border/Ms[i]
            t_star, distances = ascent(t_vec,T,σ,σ_0,t_vec_compare,η,sign_ascent,N_steps)
            delta = distances - dists[i,:]
            dists[i,:] += delta/j
            delta2 = distances - dists[i,:]
            vars[i,:] += delta.*delta2
        end
    end
    dists,vars
end

function plot_ascent_symmetric(dists,vars,Ms,N_reps = 10)
    widthCM = 10
    heightCM = 6
    f = figure(figsize=(widthCM/2.54, heightCM/2.54), dpi=72)
    ax = gca()
    tight_layout(rect = [0.1, 0.1, 1, 1])
    ax.spines["right"].set_visible(false)
    ax.spines["top"].set_visible(false)
    N_steps = length(dists[1,:])
    colors = ["navy","red","green"]
    #for (idx,M) in enumerate(Ms)
    idx = 1
    sme = sqrt.(vars[idx,:]./N_reps)
    ax.plot(dists[idx,:], label = L"\mathbf{t}^{(0)} = \mathbf{t}_1^e", color = colors[idx])
    ax.fill_between(collect(1:N_steps), dists[idx,:]+sme, dists[idx,:]-sme, alpha = 0.15)
    idx = 2
    sme = sqrt.(vars[idx,:]./N_reps)
    ax.plot(dists[idx,:], label = L"\mathbf{t}^{(0)} = \mathbf{t}_5^e", color = colors[idx])
    ax.fill_between(collect(1:N_steps), dists[idx,:]+sme, dists[idx,:]-sme, alpha = 0.15)
    idx = 3
    sme = sqrt.(vars[idx,:]./N_reps)
    ax.plot(dists[idx,:], label = L"\mathbf{t}^{(0)} = \mathbf{t}_9^e", color = colors[idx])
    ax.fill_between(collect(1:N_steps), dists[idx,:]+sme, dists[idx,:]-sme, alpha = 0.15)
    #end
    legend(fontsize = 14, frameon = false)
    ax.set_yticks([0.0,0.5,1.0])
    ax.tick_params(labelsize=14,direction = "out",top=false)
    ax.set_xlabel(L"Step $k$",fontsize = 16)
    ax.set_ylabel(L"|| \mathbf{t}^e_5-\mathbf{t}^{(k)}|| /T",rotation=90,fontsize = 16)
end

end