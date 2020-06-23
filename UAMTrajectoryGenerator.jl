########################################################
# Main file to include for generating UAM encounters
# Loads in all constants and functions
########################################################

using Distributions
using Convex
using ECOS
using DataFrames
using CSV
using Random

"""
---------------------------------------------------
Constants
---------------------------------------------------
"""

fpm2fps = 1/60
ft2m = 0.3048
m2ft = 3.28084
g = 9.81
g_ft = 32.2

λ = 1

"""
---------------------------------------------------
Types and constructors
---------------------------------------------------
"""

abstract type SAMPLER
end

abstract type UAM_TRAJECTORY
end

mutable struct TRAJECTORY
	p::Array{Float64,2}
	v::Array{Float64,2}
	a::Array{Float64,2}
end

function trajectory(;p = zeros(1,3),
					v = zeros(1,3),
					a = zeros(1,3))
	return TRAJECTORY(p, v, a)
end

"""
---------------------------------------------------
Include files for defined trajectory types
---------------------------------------------------
"""

# These files should all have a sampler, trajectory
# type, and solve_trajectory!() function
include("NominalLanding.jl")
include("VerticalDescent.jl")
include("CurvedApproach.jl")
include("NominalTakeoff.jl")
include("VerticalAscent.jl")
include("HighReconnaissance.jl")

uam_trajectories_1 = [nominal_landing(), vertical_descent(), curved_approach(),
					nominal_takeoff(), vertical_ascent()]
uam_trajectories_2 = [nominal_landing(), vertical_descent(), curved_approach(),
					nominal_takeoff(), vertical_ascent(), high_reconnaissance()]
uam_trajectories_no_takeoff = [nominal_landing(), vertical_descent(), curved_approach(),
							 high_reconnaissance()]

landing_trajectories = [NOMINAL_LANDING, VERTICAL_DESCENT, CURVED_APPROACH]
takeoff_trajectories = [NOMINAL_TAKEOFF, VERTICAL_ASCENT]

"""
---------------------------------------------------
Functions
---------------------------------------------------
"""

###################################################
# Actually generating trajectory
# (Modifies encounter object)
###################################################
function generate_trajectory!(τ::UAM_TRAJECTORY)
	optimal = false
	count = 0
	while !optimal
		if count > 20
			return false
		end
		sample_features!(τ)
		optimal = solve_trajectory!(τ)
		count += 1
	end
	return true
end

function sample_features!(τ::UAM_TRAJECTORY)
	sampler = τ.sampler
	for f in fieldnames(typeof(sampler))
		d = getfield(sampler, f)
		s = rand(d)
		setfield!(τ, f, s)
	end
end

###################################################
# General functions that operate on all UAM 
# trajectory types
###################################################
function resize!(τ::UAM_TRAJECTORY, t_length::Float64)
	num_steps = size(τ.p, 1)
	curr_times = collect(range(0, step=τ.dt, length=num_steps))
	extra_time = t_length - curr_times[end]
	extra_steps = convert(Int64, floor(extra_time/τ.dt))
	if extra_steps < 0
		τ.p = τ.p[1:end+extra_steps,:]
		τ.v = τ.v[1:end+extra_steps,:]
	elseif extra_steps > 0
		extend!(τ, t_length)
	end
end

function extend!(τ::UAM_TRAJECTORY, t_length::Float64)
	num_steps = size(τ.p, 1)
	curr_times = collect(range(0, step=τ.dt, length=num_steps))
	extra_time = t_length - curr_times[end]
	extra_steps = convert(Int64, floor(extra_time/τ.dt))
	if extra_steps ≤ 0
		println("Already $t_length s or longer.")
		return
	end

	landing = typeof(τ) in landing_trajectories
	v_extend = landing ? -τ.v[1,:] : τ.v[end,:]
	v_extend[3] = 0.0
	p_start = landing ? τ.p[1,:] : τ.p[end,:]
	new_p = zeros(extra_steps, 3)
	new_p[1,:] = p_start + v_extend*τ.dt
	for i = 1:extra_steps-1
		new_p[i+1,:] = new_p[i,:] + v_extend*τ.dt
	end

	# Update in trajectory
	if landing
		new_p = reverse(new_p, dims=1)
		τ.p = [new_p; τ.p]
		τ.v = [repeat(-v_extend, 1, extra_steps)'; τ.v]
	else
		τ.p = [τ.p; new_p]
		τ.v = [τ.v; repeat(v_extend, 1, extra_steps)']
	end
end

function generate_trajectory_file(τ::UAM_TRAJECTORY, filename::String)
	generate_trajectory!(τ)
	traj = get_trajectory(τ)
	times = collect(range(0, step = τ.dt, length = length(traj.p[:,1])))
	df = DataFrame(time_s = times, x_ft = traj.p[:,1].*m2ft, y_ft = traj.p[:,2].*m2ft, z_ft = traj.p[:,3])
	CSV.write(filename, df)
end

function generate_trajectory_file(filename::String)
	τ = uam_trajectories_2[rand(1:end)]
	generate_trajectory!(τ)
	traj = get_trajectory(τ)
	times = collect(range(0, step = τ.dt, length = length(traj.p[:,1])))
	df = DataFrame(time_s = times, x_ft = traj.p[:,1].*m2ft, y_ft = traj.p[:,2].*m2ft, z_ft = traj.p[:,3])
	CSV.write(filename, df)
end

function generate_trajectory_file(τ::UAM_TRAJECTORY, filename::String, num_trajs::Int64)
	xs = []
	ys = []
	zs = []
	times = []
	for i = 1:num_trajs
		generate_trajectory!(τ)
		traj = get_trajectory(τ)
		times = [times; collect(range(0, step = τ.dt, length = length(traj.p[:,1])))]
		xs = [xs; traj.p[:,1].*m2ft]
		ys = [ys; traj.p[:,2].*m2ft]
		zs = [zs; traj.p[:,3]]
	end
	df = DataFrame(time_s = times, x_ft = xs, y_ft = ys, z_ft = zs)
	CSV.write(filename, df)
end

function generate_trajectory_file(filename::String, num_trajs::Int64)
	τ = uam_trajectories_2[rand(1:end)]
	xs = []
	ys = []
	zs = []
	times = []
	for i = 1:num_trajs
		generate_trajectory!(τ)
		traj = get_trajectory(τ)
		times = [times; collect(range(0, step = τ.dt, length = length(traj.p[:,1])))]
		xs = [xs; traj.p[:,1].*m2ft]
		ys = [ys; traj.p[:,2].*m2ft]
		zs = [zs; traj.p[:,3]]
	end
	df = DataFrame(time_s = times, x_ft = xs, y_ft = ys, z_ft = zs)
	CSV.write(filename, df)
end

function get_trajectory(τ::UAM_TRAJECTORY)
	p = τ.p
	v = τ.v
	
	N = size(v,1)
	D = zeros(N-1,N)
	for i = 1:(N-1)
	    D[i,i] = -1
	    D[i,i+1] = 1
	end
	D = D./τ.dt
	a = D*v
	a_extra = [0.0 0.0 v[1,3]]
	a = vcat(a_extra, a)

	return TRAJECTORY(p, v, a)
end
