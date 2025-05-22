
using OrdinaryDiffEqSSPRK, OrdinaryDiffEqLowStorageRK
using Trixi
using TrixiShallowWater

### Now define the new flux-differencing formula. Here we compute only the internal fluxes.
### We will use pure FV at the left boundary for now (standard subcell limiting)...
## We use Trixi.NonConservativeSymmetric() to dispatch SKEW-SYMMETRIC terms for now...
####################################################################

# Since this is specific for this application, we dispatch with 
# nonconservative_terms::True,
# equations::ShallowWaterMultiLayerEquations2D
# NEW (SUPER) SYMMETRIZED FORMULA
@inline function Trixi.calcflux_fhat!(fhat1_L, fhat1_R, fhat2_L, fhat2_R, u,
                                      mesh::TreeMesh{2}, nonconservative_terms::Trixi.True,
                                      equations::ShallowWaterMultiLayerEquations2D,
                                      volume_flux, dg::DGSEM, element, cache)
    @unpack weights, derivative_split = dg.basis
    @unpack flux_temp_threaded, flux_nonconservative_temp_threaded = cache
    @unpack fhat_temp_threaded, fhat_nonconservative_temp_threaded, phi_threaded = cache

    volume_flux_cons, volume_flux_noncons = volume_flux

    flux_temp = flux_temp_threaded[Threads.threadid()]
    flux_noncons_temp = flux_nonconservative_temp_threaded[Threads.threadid()]

    fhat_temp = fhat_temp_threaded[Threads.threadid()]
    fhat_noncons_temp = fhat_nonconservative_temp_threaded[Threads.threadid()]
    phi = phi_threaded[Threads.threadid()]

    # The FV-form fluxes are calculated in a recursive manner, i.e.:
    # fhat_(0,1)   = w_0 * FVol_0,
    # fhat_(j,j+1) = fhat_(j-1,j) + w_j * FVol_j,   for j=1,...,N-1,
    # with the split form volume fluxes FVol_j = -2 * sum_i=0^N D_ji f*_(j,i).

    # To use the symmetry of the `volume_flux`, the split form volume flux is precalculated
    # like in `calc_volume_integral!` for the `VolumeIntegralFluxDifferencing`
    # and saved in in `flux_temp`.

    # Split form volume flux in orientation 1: x direction
    
    # First left to right:
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for j in eachnode(dg), i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and skew-symmetry of `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i + 1):nnodes(dg)
            u_node_ii = Trixi.get_node_vars(u, equations, dg, ii, j, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            Trixi.multiply_add_to_node_vars!(flux_temp, derivative_split[i, ii], flux1,
                                             equations, dg, i, j)
            Trixi.multiply_add_to_node_vars!(flux_temp, derivative_split[ii, i], flux1,
                                             equations, dg, ii, j)
            for noncons in 1:Trixi.n_nonconservative_terms(equations)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
                                                    Trixi.NonConservativeSymmetric(),
                                                    noncons)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 0.5f0 * derivative_split[i, ii],
                                                 flux1_noncons,
                                                 equations, dg, noncons, i, j)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 -0.5f0 * derivative_split[ii, i],
                                                 flux1_noncons,
                                                 equations, dg, noncons, ii, j)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat1_L[:, 1, :] .= zero(eltype(fhat1_L))
    fhat1_L[:, nnodes(dg) + 1, :] .= zero(eltype(fhat1_L))
    fhat1_R[:, 1, :] .= zero(eltype(fhat1_R))
    fhat1_R[:, nnodes(dg) + 1, :] .= zero(eltype(fhat1_R))

    fhat_temp[:, 1, :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, 1, :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for j in eachnode(dg), i in eachnode(dg)
        u_local = Trixi.get_node_vars(u, equations, dg, i, j, element)
        for noncons in 1:Trixi.n_nonconservative_terms(equations)
            Trixi.set_node_vars!(phi,
                                 volume_flux_noncons(u_local, 1, equations,
                                                     Trixi.NonConservativeLocal(), noncons),
                                 equations, dg, noncons, i, j)
        end
    end

    for j in eachnode(dg), i in 1:(nnodes(dg) - 1)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j] + weights[i] * flux_temp[v, i, j]
            fhat_temp[v, i + 1, j] = value
            fhat1_L[v, i + 1, j] = value
            fhat1_R[v, i + 1, j] = value
        end
        # Nonconservative part
        for noncons in 1:Trixi.n_nonconservative_terms(equations),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i + 1, j] = value

            fhat1_L[v, i + 1, j] = fhat1_L[v, i + 1, j] + phi[v, noncons, i, j] * value
            fhat1_R[v, i + 1, j] = fhat1_R[v, i + 1, j] +
                                   phi[v, noncons, i + 1, j] * value
        end
    end

    # New: shift the term Gamma_{(N,N-1)} to correct the flux-diff formula for skew-symmetric fluxes!
    for j in eachnode(dg)
        u_0 = Trixi.get_node_vars(u, equations, dg, 1, j, element)
        u_N = Trixi.get_node_vars(u, equations, dg, nnodes(dg), j, element)
        for noncons in 1:Trixi.n_nonconservative_terms(equations)
            phi_skew = volume_flux_noncons(u_0, u_N, 1, equations,
                                           Trixi.NonConservativeSymmetric(), noncons)

            for v in eachvariable(equations)
                fhat1_R[v, nnodes(dg), j] -= phi[v, noncons, nnodes(dg), j] * phi_skew[v] # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
            end

            # # Extra symmetrization
            # for i in nnodes(dg)-1:-1:2, v in eachvariable(equations)
            #     fhat1_L[v, i + 1, j] -= phi[v, noncons, i, j] * phi_skew[v]
            #     fhat1_R[v, i, j] -= phi[v, noncons, i, j] * phi_skew[v]
            # end
            # Extra symmetrization
            for i in nnodes(dg)-1:-1:2, v in eachvariable(equations)
                u_i = Trixi.get_node_vars(u, equations, dg, i, j, element)
                phi_skew = volume_flux_noncons(u_0, u_i, 1, equations,
                                               Trixi.NonConservativeSymmetric(), noncons)
                fhat1_L[v, i + 1, j] -= phi[v, noncons, i, j] * phi_skew[v]
                fhat1_R[v, i, j] -= phi[v, noncons, i, j] * phi_skew[v]
            end
        end
    end

    # Now right to left:
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for j in nnodes(dg):-1:1, i in nnodes(dg):-1:1
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)

        # All diagonal entries of `derivative_split` are zero. Thus, we can skip
        # the computation of the diagonal terms. In addition, we use the symmetry
        # of `volume_flux_cons` and skew-symmetry of `volume_flux_noncons` to save half of the possible two-point flux
        # computations.
        for ii in (i - 1):-1:1
            u_node_ii = Trixi.get_node_vars(u, equations, dg, ii, j, element)
            flux1 = volume_flux_cons(u_node, u_node_ii, 1, equations)
            Trixi.multiply_add_to_node_vars!(flux_temp, -derivative_split[i, ii], flux1,
                                             equations, dg, i, j)
            Trixi.multiply_add_to_node_vars!(flux_temp, -derivative_split[ii, i], flux1,
                                             equations, dg, ii, j)
            for noncons in 1:Trixi.n_nonconservative_terms(equations)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux1_noncons = volume_flux_noncons(u_node, u_node_ii, 1, equations,
                                                    Trixi.NonConservativeSymmetric(),
                                                    noncons)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 -0.5f0 * derivative_split[i, ii],
                                                 flux1_noncons,
                                                 equations, dg, noncons, i, j)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 0.5f0 * derivative_split[ii, i],
                                                 flux1_noncons,
                                                 equations, dg, noncons, ii, j)
            end
        end
    end

    # FV-form flux `fhat` in x direction
    fhat_temp[:, nnodes(dg), :] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, nnodes(dg), :] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux (already done!!)

    for j in nnodes(dg):-1:1, i in nnodes(dg):-1:2
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j] + weights[i] * flux_temp[v, i, j]
            fhat_temp[v, i - 1, j] = value
            fhat1_L[v, i, j] += value
            fhat1_R[v, i, j] += value
        end
        # Nonconservative part
        for noncons in 1:Trixi.n_nonconservative_terms(equations),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[i] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i - 1, j] = value

            fhat1_R[v, i, j] += phi[v, noncons, i, j] * value
            fhat1_L[v, i, j] += phi[v, noncons, i - 1, j] * value
        end
    end

    # New: shift the term Gamma_{(1,2)} to correct the flux-diff formula for skew-symmetric fluxes!
    for j in nnodes(dg):-1:1
        u_0 = Trixi.get_node_vars(u, equations, dg, 1, j, element)
        u_N = Trixi.get_node_vars(u, equations, dg, nnodes(dg), j, element)
        for noncons in 1:Trixi.n_nonconservative_terms(equations)
            phi_skew = volume_flux_noncons(u_0, u_N, 1, equations,
                                           Trixi.NonConservativeSymmetric(), noncons)

            for v in eachvariable(equations)
                fhat1_L[v, 2, j] += phi[v, noncons, 1, j] * phi_skew[v] # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
            end

            # Extra symmetrization
            for i in 2:nnodes(dg) - 1, v in eachvariable(equations)
                u_i = Trixi.get_node_vars(u, equations, dg, i, j, element)
                phi_skew = volume_flux_noncons(u_i, u_N, 1, equations,
                                               Trixi.NonConservativeSymmetric(), noncons)
                fhat1_R[v, i, j] += phi[v, noncons, i, j] * phi_skew[v]
                fhat1_L[v, i + 1, j] += phi[v, noncons, i, j] * phi_skew[v]
            end
        end
    end
    # Now take the average!
    fhat1_L .*= 0.5f0
    fhat1_R .*= 0.5f0

    ########

    # Split form volume flux in orientation 2: y direction

    # First do left to right
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for j in eachnode(dg), i in eachnode(dg)
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
        for jj in (j + 1):nnodes(dg)
            u_node_jj = Trixi.get_node_vars(u, equations, dg, i, jj, element)
            flux2 = volume_flux_cons(u_node, u_node_jj, 2, equations)
            Trixi.multiply_add_to_node_vars!(flux_temp, derivative_split[j, jj], flux2,
                                             equations, dg, i, j)
            Trixi.multiply_add_to_node_vars!(flux_temp, derivative_split[jj, j], flux2,
                                             equations, dg, i, jj)
            for noncons in 1:Trixi.n_nonconservative_terms(equations)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    Trixi.NonConservativeSymmetric(),
                                                    noncons)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 0.5 * derivative_split[j, jj],
                                                 flux2_noncons,
                                                 equations, dg, noncons, i, j)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 -0.5 * derivative_split[jj, j],
                                                 flux2_noncons,
                                                 equations, dg, noncons, i, jj)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat2_L[:, :, 1] .= zero(eltype(fhat2_L))
    fhat2_L[:, :, nnodes(dg) + 1] .= zero(eltype(fhat2_L))
    fhat2_R[:, :, 1] .= zero(eltype(fhat2_R))
    fhat2_R[:, :, nnodes(dg) + 1] .= zero(eltype(fhat2_R))

    fhat_temp[:, :, 1] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, 1] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux
    for j in eachnode(dg), i in eachnode(dg)
        u_local = Trixi.get_node_vars(u, equations, dg, i, j, element)
        for noncons in 1:Trixi.n_nonconservative_terms(equations)
            Trixi.set_node_vars!(phi,
                                 volume_flux_noncons(u_local, 2, equations,
                                                     Trixi.NonConservativeLocal(), noncons),
                                 equations, dg, noncons, i, j)
        end
    end

    for j in 1:(nnodes(dg) - 1), i in eachnode(dg)
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j] + weights[j] * flux_temp[v, i, j]
            fhat_temp[v, i, j + 1] = value
            fhat2_L[v, i, j + 1] = value
            fhat2_R[v, i, j + 1] = value
        end
        # Nonconservative part
        for noncons in 1:Trixi.n_nonconservative_terms(equations),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i, j + 1] = value

            fhat2_L[v, i, j + 1] = fhat2_L[v, i, j + 1] + phi[v, noncons, i, j] * value
            fhat2_R[v, i, j + 1] = fhat2_R[v, i, j + 1] +
                                   phi[v, noncons, i, j + 1] * value
        end
    end

    # New: shift the term Gamma_{(N,N-1)} to correct the flux-diff formula for skew-symmetric fluxes!
    for i in eachnode(dg)
        u_0 = Trixi.get_node_vars(u, equations, dg, i, 1, element)
        u_N = Trixi.get_node_vars(u, equations, dg, i, nnodes(dg), element)
        for noncons in 1:Trixi.n_nonconservative_terms(equations)
            phi_skew = volume_flux_noncons(u_0, u_N, 2, equations,
                                           Trixi.NonConservativeSymmetric(), noncons)

            for v in eachvariable(equations)
                fhat2_R[v, i, nnodes(dg)] -= phi[v, noncons, i, nnodes(dg)] * phi_skew[v] # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
            end

            # # Extra symmetrization
            # for j in nnodes(dg)-1:-1:2, v in eachvariable(equations)
            #     fhat2_L[v, i, j + 1] -= phi[v, noncons, i, j] * phi_skew[v]
            #     fhat2_R[v, i, j] -= phi[v, noncons, i, j] * phi_skew[v]
            # end
            # Extra symmetrization
            for j in nnodes(dg)-1:-1:2, v in eachvariable(equations)
                u_j = Trixi.get_node_vars(u, equations, dg, i, j, element)
                phi_skew = volume_flux_noncons(u_0, u_j, 2, equations,
                                               Trixi.NonConservativeSymmetric(), noncons)
                fhat2_L[v, i, j + 1] -= phi[v, noncons, i, j] * phi_skew[v]
                fhat2_R[v, i, j] -= phi[v, noncons, i, j] * phi_skew[v]
            end
        end
    end

    # Now do right to left:
    flux_temp .= zero(eltype(flux_temp))
    flux_noncons_temp .= zero(eltype(flux_noncons_temp))

    for j in nnodes(dg):-1:1, i in nnodes(dg):-1:1
        u_node = Trixi.get_node_vars(u, equations, dg, i, j, element)
        for jj in (j - 1):-1:1
            u_node_jj = Trixi.get_node_vars(u, equations, dg, i, jj, element)
            flux2 = volume_flux_cons(u_node, u_node_jj, 2, equations)
            Trixi.multiply_add_to_node_vars!(flux_temp, -derivative_split[j, jj], flux2,
                                             equations, dg, i, j)
            Trixi.multiply_add_to_node_vars!(flux_temp, -derivative_split[jj, j], flux2,
                                             equations, dg, i, jj)
            for noncons in 1:Trixi.n_nonconservative_terms(equations)
                # We multiply by 0.5 because that is done in other parts of Trixi
                flux2_noncons = volume_flux_noncons(u_node, u_node_jj, 2, equations,
                                                    Trixi.NonConservativeSymmetric(),
                                                    noncons)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 -0.5 * derivative_split[j, jj],
                                                 flux2_noncons,
                                                 equations, dg, noncons, i, j)
                Trixi.multiply_add_to_node_vars!(flux_noncons_temp,
                                                 0.5 * derivative_split[jj, j],
                                                 flux2_noncons,
                                                 equations, dg, noncons, i, jj)
            end
        end
    end

    # FV-form flux `fhat` in y direction
    fhat_temp[:, :, nnodes(dg)] .= zero(eltype(fhat1_L))
    fhat_noncons_temp[:, :, :, nnodes(dg)] .= zero(eltype(fhat1_L))

    # Compute local contribution to non-conservative flux (already done!!!)

    for j in nnodes(dg):-1:2, i in nnodes(dg):-1:1
        # Conservative part
        for v in eachvariable(equations)
            value = fhat_temp[v, i, j] + weights[j] * flux_temp[v, i, j]
            fhat_temp[v, i, j - 1] = value
            fhat2_L[v, i, j] += value
            fhat2_R[v, i, j] += value
        end
        # Nonconservative part
        for noncons in 1:Trixi.n_nonconservative_terms(equations),
            v in eachvariable(equations)

            value = fhat_noncons_temp[v, noncons, i, j] +
                    weights[j] * flux_noncons_temp[v, noncons, i, j]
            fhat_noncons_temp[v, noncons, i, j - 1] = value

            fhat2_R[v, i, j] += phi[v, noncons, i, j] * value
            fhat2_L[v, i, j] += phi[v, noncons, i, j - 1] * value
        end
    end

    # New: shift the term Gamma_{(1,2)} to correct the flux-diff formula for skew-symmetric fluxes!
    for i in nnodes(dg):-1:1
        u_0 = Trixi.get_node_vars(u, equations, dg, i, 1, element)
        u_N = Trixi.get_node_vars(u, equations, dg, i, nnodes(dg), element)
        for noncons in 1:Trixi.n_nonconservative_terms(equations)
            phi_skew = volume_flux_noncons(u_0, u_N, 2, equations,
                                           Trixi.NonConservativeSymmetric(), noncons)

            for v in eachvariable(equations)
                fhat2_L[v, i, 2] += phi[v, noncons, i, 1] * phi_skew[v] # The factor of 2 is missing cause Trixi multiplies all the non-cons terms with 0.5
            end

            # Extra symmetrization
            for j in 2:nnodes(dg) - 1, v in eachvariable(equations)
                u_j = Trixi.get_node_vars(u, equations, dg, i, j, element)
                phi_skew = volume_flux_noncons(u_j, u_N, 2, equations,
                                               Trixi.NonConservativeSymmetric(), noncons)
                fhat2_R[v, i, j] += phi[v, noncons, i, j] * phi_skew[v]
                fhat2_L[v, i, j + 1] += phi[v, noncons, i, j] * phi_skew[v]
            end
        end
    end
    fhat2_L .*= 0.5f0
    fhat2_R .*= 0.5f0
    #######

    return nothing
end

################################
# Define missing functions for ShallowWaterMultiLayerEquations2D
Trixi.n_nonconservative_terms(::ShallowWaterMultiLayerEquations2D) = 1

@inline function TrixiShallowWater.flux_nonconservative_ersing_etal(u_ll,
                                                                    orientation::Integer,
                                                                    equations::ShallowWaterMultiLayerEquations2D,
                                                                    ::Trixi.NonConservativeLocal,
                                                                    noncons)
    # Pull the necessary left and right state information
    h_ll = waterheight(u_ll, equations)

    g = equations.gravity

    # Initialize flux vector
    f = zero(Trixi.MVector{3 * nlayers(equations) + 1, real(equations)})

    # Compute the nonconservative flux in each layer
    # where f_hv[i] = g * h[i] * (b + ∑h[k] + ∑σ[k] * h[k])_x and σ[k] = ρ[k] / ρ[i] denotes the 
    # density ratio of different layers
    for i in eachlayer(equations)
        f_hv = g * h_ll[i]

        if orientation == 1
            setindex!(f, f_hv, i + nlayers(equations))
        else # orientation == 2
            setindex!(f, f_hv, i + 2 * nlayers(equations))
        end
    end

    return SVector(f)
end

@inline function TrixiShallowWater.flux_nonconservative_ersing_etal(u_ll, u_rr,
                                                                    orientation::Integer,
                                                                    equations::ShallowWaterMultiLayerEquations2D,
                                                                    ::Trixi.NonConservativeSymmetric,
                                                                    noncons)
    # Pull the necessary left and right state information
    h_ll = waterheight(u_ll, equations)
    h_rr = waterheight(u_rr, equations)
    b_rr = u_rr[end]
    b_ll = u_ll[end]

    # Compute the jumps
    h_jump = h_rr - h_ll
    b_jump = b_rr - b_ll
    g = equations.gravity

    # Initialize flux vector
    f = zero(Trixi.MVector{3 * nlayers(equations) + 1, real(equations)})

    # Compute the nonconservative flux in each layer
    # where f_hv[i] = g * h[i] * (b + ∑h[k] + ∑σ[k] * h[k])_x and σ[k] = ρ[k] / ρ[i] denotes the 
    # density ratio of different layers
    for i in eachlayer(equations)
        f_hv = b_jump
        for j in eachlayer(equations)
            if j < i
                f_hv += (equations.rhos[j] / equations.rhos[i] * h_jump[j])
            else # (i<j<nlayers) nonconservative formulation of the pressure
                f_hv += h_jump[j]
            end
        end

        if orientation == 1
            setindex!(f, f_hv, i + nlayers(equations))
        else # orientation == 2
            setindex!(f, f_hv, i + 2 * nlayers(equations))
        end
    end

    return SVector(f)
end

@inline function Trixi.flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                                       equations::ShallowWaterMultiLayerEquations2D)
    # Pull the necessary left and right state information
    h_ll = u_ll[1]
    b_avg = u_rr[4] + u_ll[4] # omiting 0.5 bla bla

    # Bottom gradient nonconservative term: (0, g h b_x, g h b_y, 0)
    if orientation == 1
        f = SVector(0, equations.gravity * h_ll * b_avg, 0, 0)
    else # orientation == 2
        f = SVector(0, 0, equations.gravity * h_ll * b_avg, 0)
    end
    return f
end

@inline function Trixi.flux_nonconservative_wintermeyer_etal(u_ll, orientation::Integer,
                                                            equations::ShallowWaterMultiLayerEquations2D,
                                                            ::Trixi.NonConservativeLocal, noncons)
    # Pull the necessary left and right state information
    h_ll = u_ll[1]

    # Bottom gradient nonconservative term: (0, g h b_x, g h b_y, 0)
    if orientation == 1
        f = SVector(0, equations.gravity * h_ll, 0, 0)
    else # orientation == 2
        f = SVector(0, 0, equations.gravity * h_ll, 0)
    end
    return f
end

@inline function Trixi.flux_nonconservative_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                                            equations::ShallowWaterMultiLayerEquations2D,
                                                            ::Trixi.NonConservativeSymmetric, noncons)
    b_avg = u_rr[4] + u_ll[4] # omiting 0.5 bla bla

    # Bottom gradient nonconservative term: (0, g h b_x, g h b_y, 0)
    if orientation == 1
        f = SVector(0, b_avg, 0, 0)
    else # orientation == 2
        f = SVector(0, 0, b_avg, 0)
    end
    return f
end

@inline function Trixi.flux_wintermeyer_etal(u_ll, u_rr, orientation::Integer,
                                       equations::ShallowWaterMultiLayerEquations2D)
    # Unpack left and right state
    h_ll, h_v1_ll, h_v2_ll, _ = u_ll
    h_rr, h_v1_rr, h_v2_rr, _ = u_rr

    # Get the velocities on either side
    inv_h_ll = 1 / h_ll
    v1_ll = inv_h_ll * h_v1_ll
    v2_ll = inv_h_ll * h_v2_ll
    inv_h_rr = 1 / h_rr
    v1_rr = inv_h_rr * h_v1_rr
    v2_rr = inv_h_rr * h_v2_rr

    # Average each factor of products in flux
    v1_avg = 0.5f0 * (v1_ll + v1_rr)
    v2_avg = 0.5f0 * (v2_ll + v2_rr)
    p_avg = 0.5f0 * equations.gravity * h_ll * h_rr

    # Calculate fluxes depending on orientation
    if orientation == 1
        f1 = 0.5f0 * (h_v1_ll + h_v1_rr)
        f2 = f1 * v1_avg + p_avg
        f3 = f1 * v2_avg
    else
        f1 = 0.5f0 * (h_v2_ll + h_v2_rr)
        f2 = f1 * v1_avg
        f3 = f1 * v2_avg + p_avg
    end

    return SVector(f1, f2, f3, 0)
end

###############################################################################
# Semidiscretization of the multilayer shallow water equations with a bottom topography function
# to test well-balancedness

equations = ShallowWaterMultiLayerEquations2D(gravity = 9.81, H0 = 0.45,
                                              rhos = (1.0))

function initial_condition_blast_wave(x, t, equations::ShallowWaterMultiLayerEquations2D)
    # Set up polar coordinates
    RealT = eltype(x)
    inicenter = SVector(convert(RealT, 2.0), convert(RealT, 2.0))
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)

    # Calculate primitive variables5
    H = r > 0.5f0 ? 2.0f0 : 4.0f0
    v1 = r > 0.5f0 ? zero(RealT) : convert(RealT, 0.1882) * cos_phi
    v2 = r > 0.5f0 ? zero(RealT) : convert(RealT, 0.1882) * sin_phi
    b = 0.2 + 0.2 * cos(pi * (x[1] + x[2])) # by default assume there is no bottom topography

    return prim2cons(SVector(H, v1, v2, b), equations)
end

function total_waterheight(u, equations::ShallowWaterMultiLayerEquations2D)
    prim = cons2prim(u, equations)
    return prim[1]
end

@inline function Trixi.gradient_conservative(::typeof(total_waterheight),
                                       u, equations::ShallowWaterMultiLayerEquations2D)
    return SVector(1, 0, 0, 1)
end

function Trixi.isvalid(u, equations::ShallowWaterMultiLayerEquations2D)
    return u[1] > 0
end

initial_condition = initial_condition_blast_wave

###############################################################################
# Get the DG approximation space

polydeg = 4
volume_flux = (flux_ersing_etal, TrixiShallowWater.flux_nonconservative_ersing_etal)
surface_flux = (FluxHydrostaticReconstruction(FluxPlusDissipation(flux_ersing_etal, DissipationLocalLaxFriedrichs()), hydrostatic_reconstruction_ersing_etal), 
                FluxHydrostaticReconstruction(flux_nonconservative_ersing_etal, hydrostatic_reconstruction_ersing_etal))
# surface_flux = (FluxPlusDissipation(flux_ersing_etal, DissipationLocalLaxFriedrichs()), TrixiShallowWater.flux_nonconservative_ersing_etal)

# volume_flux = (Trixi.flux_wintermeyer_etal, Trixi.flux_nonconservative_wintermeyer_etal)
# surface_flux = (FluxPlusDissipation(Trixi.flux_wintermeyer_etal, DissipationLocalLaxFriedrichs()), Trixi.flux_nonconservative_wintermeyer_etal)

# volume_flux = (Trixi.flux_wintermeyer_etal, Trixi.flux_nonconservative_wintermeyer_etal)
# surface_flux = (FluxHydrostaticReconstruction(FluxPlusDissipation(Trixi.flux_wintermeyer_etal, DissipationLocalLaxFriedrichs()), hydrostatic_reconstruction_ersing_etal), 
#                 FluxHydrostaticReconstruction(Trixi.flux_nonconservative_wintermeyer_etal, hydrostatic_reconstruction_ersing_etal))

basis = LobattoLegendreBasis(polydeg)
limiter_idp = SubcellLimiterIDP(equations, basis;
                                #positivity_variables_cons = ["h1"], 
                                local_twosided_variables_cons = ["h1"],
                                #local_onesided_variables_nonlinear = [(total_waterheight, min), (total_waterheight, max)],
                                )
volume_integral = VolumeIntegralSubcellLimiting(limiter_idp;
                                                volume_flux_dg = volume_flux,
                                                volume_flux_fv = surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

###############################################################################
# Get the TreeMesh and setup a periodic mesh

coordinates_min = (0.0, 0.0)
coordinates_max = (4.0, 4.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = 5,
                n_cells_max = 10_000,
                periodicity = true)

# Create the semi discretization object
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)

###############################################################################
# ODE solver

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     analysis_polydeg = polydeg)

stepsize_callback = StepsizeCallback(cfl = 0.4)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 20,
                                     save_initial_solution = true,
                                     save_final_solution = true)

callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

stage_callbacks = (SubcellLimiterIDPCorrection(),)

sol = Trixi.solve(ode, Trixi.SimpleSSPRK33(stage_callbacks = stage_callbacks);
                  dt = 1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                  ode_default_options()...,
                  callback = callbacks);
