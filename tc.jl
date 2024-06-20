"""
Usage
include("tc.jl")
tc.setup()
tc.derive()
soln = tc.evolve();
tc.plot_dynamics(soln);
OR
include("tc.jl")
tc.run()
# SEE ALSO cumulants/qojulia/dicke/tc.jl (basically same thing, used to test Nm
# dependence)
"""
module tc
    if !@isdefined(QuantumCumulants)
        using QuantumCumulants, ModelingToolkit, OrdinaryDiffEq, Plots, DelimitedFiles, Printf
    end
    function run(; order=1, N=100000, tend=300)
    """Single run with default options / parameters"""
        setup()
        derive(order=order)
        u0 = get_u0(a=0.13338429247292602,
                    sp=-0.0004712602672110967,
                    p_up=2.2208628880360237e-07)
        sol=evolve(N0=N, tend=tend, u0=u0)
        plot_dynamics(sol)
    end
    """
    Setup Hilbert spaces, Hamiltonion, rates and initial operators (to calculate EoMs)
    for the Tavis-Cummings model.
    """
    function setup(; sr=false, rwa=true)
        global hc = FockSpace(:cavity);
        global ha = NLevelSpace(:satellite,2);
        global h = hc ⊗  ha;
        global RWA = rwa
        global EV_TO_FS = 0.658212  
        
        @cnumbers Ω N g ω ϵ κ Γu Γd Γz;
    
        i = Index(h,:i,N,ha);
        j = Index(h,:j,N,ha);
        k = Index(h,:k,N,ha);
        
        global a = Destroy(h, :a)
        σzi = σ(2,2,i)-σ(1,1,i);
        
        global H = ω*a'*a + ∑((ϵ//2)*σzi + g*(a*σ(2,1,i) + a'*σ(1,2,i)),i);
        if !rwa
            H = H + ∑(g*(a'*σ(2,1,i) + a*σ(1,2,i)),i);
        end
        global J = [a, σ(1,2,i), σ(2,1,i), σzi];
        global rates = [κ, Γd, Γu, Γz];
        global ops = [a'*a,
               σ(2,1,j)*σ(1,2,j),
              ];
        println("Defined H, rates and starting operators")
    end

    function σ(x,y,z)
        return IndexedOperator(Transition(h,:σ,x,y,2),z)
    end
   
    """
    Derive a complete set of equations of motion to a specified order of cumulant
    expansion. Scales the equations so that sums over identical spins are replaced 
    by factors of N. Keeps symmetry breaking terms that vanish for symemtric 
    initial conditions (see evolve_loop).
    """
    function derive(; order=1)
        global H, J, rates, ops, ha
        if !@isdefined(H)
            println("Hamiltonian not defined - run tc.setup() first!")
            return
        end
        global ORDER=order
        if order==1
            global h, ha
            @cnumbers N
            j = Index(h,:j,N,ha); # need an indexed operator!!
            global ops = [a,  σ(2,1,j)];
        end
        global eqs = meanfield(ops,H,J;rates=rates,order=order);
        global eqs_complete = complete(eqs);
        println("Derived closed set of equations at order ", order, "...")
        global eqs_scale = scale(eqs_complete;ha); # must be called after complete
        println("Replaced sums with factors of Nm...")
        @named sys = ODESystem(eqs_scale);
        global sys = sys;
        println("Setup ODE system 'sys' for problem with scaled equations 'eqs_scale'")
        println("View these with tc.sys, tc.eqs_scale, tc.sys.parameters etc.")
    end
    
    function get_p0(; N0=100000)
        @cnumbers Ω N g ω ϵ κ Γu Γd Γz; # set of symbols for Hamiltonian / rate parameters
        global Nm, gn
        Nm = N0
        gn = 0.45
        κ0 = 0.05
        ω0 = 0.2
        ϵ0 = 0.1
        g0 = gn/sqrt(N0)
        Γu0 = 0.06
        Γd0 = 0.05
        Γz0 = 0.01
        p0 = (ω=>ω0, ϵ=>ϵ0, g=>g0, N=>N0, κ=>κ0, Γu=>Γu0, Γd=>Γd0, Γz=>Γz0)
        return p0
    end

    function get_u0(; a=0.00041856069223251047,
            sp=-0.00047413803847996827,
            p_up=2.2480693001591945e-07)
        global ORDER, sys, eqs_scale, theta # equations from derive()
        u0 = zeros(ComplexF64, length(eqs_scale)); # initial state
        if ORDER==1
            u0[1] = a # <a>
            u0[2] = sp # <sigma^+>
            u0[3] = p_up # <sigma^+sigma^-?
            println("Set-up initial conditions with", sys.states[1], " = ", u0[1],
                    " ", sys.states[2], " = ", u0[2],
                    " ", sys.states[3], " = ", u0[3])
            return u0
        end
        if ORDER==2
            println("NOT IMPLEMENTED")
            return u0
        end
    end
    
    function evolve(; tend=nothing, u0=nothing, p0=nothing, N0=100000)
        if tend==nothing
            tend = 300
        else
            tend = tend
        end
        if p0==nothing
            p0 = get_p0(N0=N0)
        end
        if u0==nothing
            u0 = get_u0()
        end
        global sys
        if !@isdefined(sys)
            println("No equations calculated - run tc.derive() first!")
            return
        end
        println(u0)
        println(p0)
        prob = ODEProblem(sys,u0,(0.0,tend),p0);
        sol = solve(prob, Tsit5(), reltol=1e-10, abstol=1e-10);
        println("Calculated dynamics to tend = ", tend)
        return sol
    end

    function plot_dynamics(sol)
        ts = sol.t 
        if ORDER == 1
            ns = real.(sol[a] .* conj(sol[a]))
        else
            ns = real.(sol[a'*a])
        end
        #zs = real.(2*sol[σ(2,2,1)].-1)
        ps = real.(sol[σ(2,2,1)])
        p1 = plot(ts, ns, xlabel="t", ylabel="⟨a⁺a⟩", legend = false)
        p2 = plot(ts, ps, xlabel="t", ylabel="⟨σ⁺σ⁻⟩", legend = false)
        results = [ts, ns, ps]
        writedlm(string("data/julia/", "gn", gn, "N", Nm, "_o", ORDER, ".csv"), results)
        plotd = plot(p1, p2, layout=(1,2), size=(800,400))
        savefig(plotd,"figures/julia.png")
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
end
