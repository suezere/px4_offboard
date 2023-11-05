from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import *
import scipy.linalg
import numpy as np

def drone_model() -> AcadosModel:
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Translational_drone"

    # Quadrotor intrinsic parameters
    # m = 1.0  # kg
    # J = np.array([.03, .03, .06])  # N m s^2 = kg m^2

    # Length of motor to CoG segment
    # length = 0.47 / 2  # m

    # mujoco parameters
    # m =  0.75 # m=27g
    # J = np.array([0.53, 0.49, 0.98])
    # length = 0.046

    hover_thrust = 0.76


    # constants
    g = 9.81 # m/s^2

    ## CasAdi Model
    # set up states and controls
    px = MX.sym("px")
    py = MX.sym("py")
    pz = MX.sym("pz")
    qw = MX.sym("qw")
    qx = MX.sym("qx")
    qy = MX.sym("qy")
    qz = MX.sym("qz")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    vz = MX.sym("vz")
    x  = vertcat(px, py, pz, qw, qx, qy, qz, vx, vy, vz)

    # controls
    T  = MX.sym("T")
    wx = MX.sym("qw")
    wy = MX.sym("qx")
    wz = MX.sym("qy")
    u = vertcat(T, wx, wy, wz)

    # xdot
    pxdot = MX.sym("pxdot")
    pydot = MX.sym("pydot")
    pzdot = MX.sym("pzdot")
    qwdot = MX.sym("qwdot")
    qxdot = MX.sym("qxdot")
    qydot = MX.sym("qydot")
    qzdot = MX.sym("qzdot")
    vxdot = MX.sym("vxdot")
    vydot = MX.sym("vydot")
    vzdot = MX.sym("vzdot")
    xdot  = vertcat(pxdot, pydot, pzdot, qwdot, qxdot, qydot, qzdot, vxdot, vydot, vzdot)

    # algebraic variables 
    z = vertcat([])

    # parameters
    p = vertcat([])

    # dynamics
    f_expl = vertcat(
        vx,
        vy,
        vz,
        0.5 * ( - wx * qx - wy * qy - wz * qz),
        0.5 * (   wx * qw + wz * qy - wy * qz),
        0.5 * (   wy * qw - wz * qx + wx * qz),
        0.5 * (   wz * qw + wy * qx - wx * qy),
        2 * ( qw * qy + qx * qz ) * T / hover_thrust * g,
        2 * ( qy * qz - qw * qx ) * T / hover_thrust * g,
        ( ( 1 - 2 * qx * qx - 2 * qy * qy ) * T ) / hover_thrust * g - g 
    )
    
    # model.phi_min = -80 * np.pi / 180
    # model.phi_max =  80 * np.pi / 180

    # model.theta_min = -80 * np.pi / 180
    # model.theta_max =  80 * np.pi / 180

    # input bounds
    model.thrust_max = 0.9  # 90 % of max_thrust (max_thrust = 57g in research papers) ----- ( max_thrsut = 46g when tested) 
    model.thrust_min = 0.1 
    # model.thrust_max = 0.9 * ((46.3e-3 * g)) # 90 % of max_thrust (max_thrust = 57g in research papers) ----- ( max_thrsut = 46g when tested) 
    # model.thrust_min = 0.1 * model.thrust_max

    # model.torque_max = 1 / 2 * model.thrust_max * length # divided by 2 since we only have 2 propellers in a planar quadrotor
    # model.torque_max = 0.1 * model.torque_max # keeping 10% margin for steering torque. This is done because the torque_max 
    #                                           # is the maximum torque that can be given around any one axis. But, we are going to
    #                                           # limit the torque greatly.
    # model.torque_min = - model.torque_max

    # define initial condition
    model.x0 = np.array([0.0, 0.0, 0.31642, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # define model struct
    params = types.SimpleNamespace()
    params.hover_thrust = hover_thrust
    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params

    return model

def acados_settings(N):

    # create OCP object to formulate the optimization
    ocp = AcadosOcp()

    # export model
    model = drone_model()

    # constants
    g = 9.81 # m/s^2
    Tf = 1.0  # prediction horizon
    Ts = Tf / N  # sampling time[s]

    # define acados ODE 
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # dimensions 
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu
    ny_e = nx 

    # discretization 
    ocp.dims.N = N
    
    # set cost 
    Q = np.eye(nx)
    Q[0][0] = 1000  # weight of px
    Q[1][1] = 1000  # weight of py
    Q[2][2] = 1000  # weight of pz
    Q[3][3] = 10  # weight of qw
    Q[4][4] = 10  # weight of qx
    Q[5][5] = 10  # weight of qy
    Q[6][6] = 10  # weight of qz
    Q[7][7] = 10 # weight of vx
    Q[8][8] = 10  # weight of vy
    Q[9][9] = 10 # weight of vz

    R = np.eye(nu)
    R[0][0] = 100  # weight of Thrust
    R[1][1] = 100 # weight of wx
    R[2][2] = 100  # weight of wy
    R[3][3] = 100  # weight of wz

    Qe = np.eye(nx)
    Qe[0][0] = 100  # terminal weight of px
    Qe[1][1] = 100  # terminal weight of py
    Qe[2][2] = 100  # terminal weight of pz
    Qe[3][3] = 10  # terminal weight of qw
    Qe[4][4] = 10  # terminal weight of qx
    Qe[5][5] = 10  # terminal weight of qy
    Qe[6][6] = 10  # terminal weight of qz
    Qe[7][7] = 10  # terminal weight of vx
    Qe[8][8] = 10  # terminal weight of vy
    Qe[9][9] = 10  # terminal weight of vz

    ocp.cost.cost_type   = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    
    ocp.cost.W   = scipy.linalg.block_diag(Q,R)
    ocp.cost.W_e = Qe

    Vx = np.zeros((ny,nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[-4:,-4:] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e
    
    # Initial reference trajectory (will be overwritten during the simulation)
    x_ref = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ocp.cost.yref   = np.concatenate((x_ref, np.array([model.params.hover_thrust * g, 0.0, 0.0, 0.0])))

    ocp.cost.yref_e = x_ref

    # set constraints on thrust and angular velocities
    ocp.constraints.lbu   = np.array([model.thrust_min, -40*np.pi, -40*np.pi, -10*np.pi])
    ocp.constraints.ubu   = np.array([model.thrust_max,  40*np.pi,  40*np.pi,  10*np.pi])
    ocp.constraints.idxbu = np.array([0,1,2,3])

    
    # ocp.constraints.lbx     = np.array([-1, -1, -1])
    # ocp.constraints.ubx     = np.array([1, 1, 1])
    # ocp.constraints.idxbx   = np.array([7, 8, 9])

    '''
    ocp.constraints.lbx = np.array([-15.0, -15.0, -15.0]) # lower bounds on the velocity states
    ocp.constraints.ubx = np.array([ 15.0,  15.0,  15.0]) # upper bounds on the velocity states
    ocp.constraints.idxbx = np.array([3, 4, 5])
    '''

    # set initial condition
    ocp.constraints.x0 = model.x0

    # set QP solver and integration
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.nlp_solver_max_iter = 400
    ocp.solver_options.tol = 1e-4

    # create ocp solver 
    acados_solver = AcadosOcpSolver(ocp, json_file=(model_ac.name + "_" + "acados_ocp.json"))

    return model, acados_solver