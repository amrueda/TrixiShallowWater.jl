
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiShallowWater

# We first define an IDP limiter that selects a random limiting factor for each
# point of the domain

"""
    SubcellLimiterRandomIDPCorrection()

Perform A RANDOM antidiffusive correction stage for the a posteriori IDP limiter [`SubcellLimiterIDP`](@ref)
called with [`VolumeIntegralSubcellLimiting`](@ref).

!!! note
    This callback and the actual limiter [`SubcellLimiterIDP`](@ref) only work together.
    This is not a replacement but a necessary addition.
"""
struct SubcellLimiterRandomIDPCorrection end

function (limiter!::SubcellLimiterRandomIDPCorrection)(u_ode,
                                                       integrator::Trixi.SimpleIntegratorSSP,
                                                       stage)
    semi = integrator.p
    limiter!(u_ode, semi, integrator.t, integrator.dt, semi.solver.volume_integral)
end

function (limiter!::SubcellLimiterRandomIDPCorrection)(u_ode, semi, t, dt,
                                                       volume_integral::VolumeIntegralSubcellLimiting)
    Trixi.@trixi_timeit Trixi.timer() "a posteriori limiter" limiter!(u_ode, semi, t, dt,
                                                                      volume_integral.limiter)
end

function (limiter!::SubcellLimiterIDPCorrection)(u_ode, semi, t, dt,
                                                 limiter::SubcellLimiterIDP)
    mesh, equations, solver, cache = Trixi.mesh_equations_solver_cache(semi)

    u = Trixi.wrap_array(u_ode, mesh, equations, solver, cache)

    # Calculate blending factor alpha in [0,1]
    # f_ij = alpha_ij * f^(FV)_ij + (1 - alpha_ij) * f^(DG)_ij
    #      = f^(FV)_ij + (1 - alpha_ij) * f^(antidiffusive)_ij
    Trixi.@trixi_timeit Trixi.timer() "blending factors" solver.volume_integral.limiter(u,
                                                                                        semi,
                                                                                        solver,
                                                                                        t,
                                                                                        dt;
                                                                                        random_factors = Trixi.True())

    Trixi.perform_idp_correction!(u, dt, mesh, equations, solver, cache)

    return nothing
end

# Compute random blending factors
function (limiter::SubcellLimiterIDP)(u::AbstractArray{<:Any, 4}, semi, dg::DGSEM, t,
                                      dt;
                                      random_factors::Trixi.True,
                                      kwargs...)

    # Calculate alpha1 and alpha2
    @unpack alpha1, alpha2 = limiter.cache.subcell_limiter_coefficients
    Trixi.@threaded for element in eachelement(dg, semi.cache)
        for j in eachnode(dg), i in 2:nnodes(dg)
            alpha1[i, j, element] = rand()
        end
        for j in 2:nnodes(dg), i in eachnode(dg)
            alpha2[i, j, element] = rand()
        end
        alpha1[1, :, element] .= zero(eltype(alpha1))
        alpha1[nnodes(dg) + 1, :, element] .= zero(eltype(alpha1))
        alpha2[:, 1, element] .= zero(eltype(alpha2))
        alpha2[:, nnodes(dg) + 1, element] .= zero(eltype(alpha2))
    end

    return nothing
end

init_callback(limiter!::SubcellLimiterRandomIDPCorrection, semi) = nothing

finalize_callback(limiter!::SubcellLimiterRandomIDPCorrection, semi) = nothing

###############################################################################
# Semidiscretization of the multilayer shallow water equations with a bottom topography function
# to test well-balancedness

equations = ShallowWaterMultiLayerEquations2D(gravity = 9.81, H0 = 0.45,
                                              rhos = (1.0))

# An initial condition with constant total water height, zero velocities and a bottom topography to
# test well-balancedness
function initial_condition_well_balanced(x, t, equations::ShallowWaterMultiLayerEquations2D)
    H = SVector(0.45)
    v1 = zero(H)
    v2 = zero(H)
    b = (((x[1] - 0.5)^2 + (x[2] - 0.5)^2) < 0.04 ?
         0.2 * (cos(4 * pi * sqrt((x[1] - 0.5)^2 + (x[2] -
                                                    0.5)^2)) + 1) : 0.0)

    return prim2cons(SVector(H..., v1..., v2..., b),
                     equations)
end

initial_condition = initial_condition_well_balanced

###############################################################################
# Get the DG approximation space

volume_flux = (flux_ersing_etal, flux_nonconservative_ersing_etal)
surface_flux = (flux_ersing_etal, flux_nonconservative_ersing_etal)
basis = LobattoLegendreBasis(3)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                positivity_variables_cons = ["h1"],)
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 3,
                n_cells_max = 10_000,
                periodicity = true)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 10.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 1000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (lake_at_rest_error,))

stepsize_callback = StepsizeCallback(cfl = 1.0)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 1000,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterRandomIDPCorrection(),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks);
