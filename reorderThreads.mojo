from std.gpu.host import DeviceContext
from layout import TileTensor
from layout.tile_layout import Layout,row_major,TensorLayout
from std.python import Python, PythonObject
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from std.collections import InlineArray
from src.lbm import SOLID_NODE,FLUID_NODE,LBM_Grid,get_D2Q9,set_outer_walls,calculate_rho_and_velocity
from src.utils import Vector,ContextTileTensor
from src.lbm.variations.part_1.reorderThreads import LBM_kernel

comptime float_dtype = DType.float32
comptime int_dtype = DType.int32
comptime float_scalar = Scalar[float_dtype]
comptime D2Q9 = get_D2Q9[DType.float32,DType.int32]()
comptime D,Q = (2,9)
comptime N = 101
comptime L = 1.
comptime dx = L/float_scalar(N-1)
comptime (nx,ny,nz) = (N,N,1)
comptime num_points = nx*ny*nz

comptime tile_size = 1
comptime grid = LBM_Grid[D2Q9,nx,ny,nz,tile_size](dx)

comptime BLOCK_SHAPE = grid.BLOCK_SHAPE
comptime GRID_DIM = grid.GRID_DIM

comptime flag_layout = row_major[nx,ny,nz]()
comptime f_layout = row_major[Q,nx,ny,nz]()
comptime bc_layout = row_major[nx,ny,nz,D+1]()
comptime density_layout = row_major[nx,ny,nz]()
comptime velocity_layout = row_major[D,nx,ny,nz]()

comptime all_slice = slice(None,None,None)

def main() raises:
    print(GRID_DIM)
    print(BLOCK_SHAPE)

    U_phs:float_scalar = 1.
    U:float_scalar = 0.05
    viscosity:float_scalar = 1/100.
    dt = dx*U/U_phs 
    Re = 1/viscosity
    L_lat:float_scalar = N
    v_lat = U*L_lat/Re
    tau = v_lat/(1/3.) +0.5
    print('Tau {}'.format(tau))

    ctx = DeviceContext()
    
    flags = ContextTileTensor[DType.uint8](ctx,flag_layout)
    bc = ContextTileTensor[float_dtype](ctx,bc_layout)
    f = ContextTileTensor[float_dtype](ctx,f_layout)
    f_out = ContextTileTensor[float_dtype](ctx,f_layout)

    u = ContextTileTensor[float_dtype](ctx,velocity_layout)
    rho = ContextTileTensor[float_dtype](ctx,density_layout)

    # Set up

    f.fill(1./Float32(Q))
    f_out.fill(1./Float32(Q))
    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'+Y',SOLID_NODE,[U,0],1.)
    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'-Y',SOLID_NODE,[0,0],1.)
    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'-X',SOLID_NODE,[0,0],1.)
    set_outer_walls[grid,flag_layout,bc_layout](flags.cpu(),bc.cpu(),'+X',SOLID_NODE,[0,0],1.)

    ctx.synchronize()
    # Copy To GPU()
    _ = flags.gpu()
    _ = bc.gpu()
    _ = f.gpu()
    _ = f_out.gpu()

    ctx.synchronize()
    #Compile Functions
    LBM_func = ctx.compile_function[LBM_kernel[grid,f_layout,bc_layout,flag_layout],LBM_kernel[grid,f_layout,bc_layout,flag_layout]]()
    calc_rho_and_u_gpu = ctx.compile_function[calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout],calculate_rho_and_velocity[grid,f_layout,density_layout,velocity_layout]]()
 
    ctx.synchronize()

    # Run Simulation
    for t in range(10_000):
        ctx.enqueue_function(LBM_func,f_out.gpu(),f.gpu(),bc.gpu(),flags.gpu(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        ctx.enqueue_function(LBM_func,f.gpu(),f_out.gpu(),bc.gpu(),flags.gpu(),1/tau,grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
        if t % 1000 == 0 :
            pass
            ctx.synchronize()
            ctx.enqueue_function(calc_rho_and_u_gpu,f.gpu(),rho.gpu(),u.gpu(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)
            ctx.synchronize()
            u_np = u.buffer_to_numpy()/U
            print('step = {} max ={} avg = {}'.format(t,u_np.max(),u_np.mean()))
    ctx.synchronize()
    # Get Final U and rho
    ctx.enqueue_function(calc_rho_and_u_gpu,f.gpu(),rho.gpu(),u.gpu(),grid_dim = GRID_DIM,block_dim = BLOCK_SHAPE)

    ctx.synchronize()
    
    np = Python.import_module('numpy')
    pd = Python.import_module('pandas')
    pv = Python.import_module('pyvista')
    plt = Python.import_module('matplotlib.pyplot')
    ctypes = Python.import_module("ctypes")

    u_np = (u.buffer_to_numpy()/U).reshape(D,nx,ny,nz)
    print('step = {} max ={} avg = {}'.format(0,u_np.max(),u_np.mean()) )

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    m = np.meshgrid(x, y)
    xx,yy = m[0],m[1]
    pv_mesh = pv.StructuredGrid(xx, yy, np.zeros_like(xx))
    
    u_plot = u_np[0,all_slice,all_slice,all_slice]
    v_plot = u_np[1,all_slice,all_slice,all_slice]

    u_mag = np.sqrt(u_plot**2 + v_plot**2)
    pv_mesh.point_data['U_mag'] = u_mag.ravel()
    pv_mesh.point_data['U velocity'] = u_plot.ravel()
    pv_mesh.point_data['V velocity'] = v_plot.ravel()
    
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh,scalars ='U_mag',show_edges = False, cmap= 'jet',clim = [0,1],nan_color='white',)
    plotter.view_xy()
    plotter.show() # screenshot = 'LDC_Re100.png'
   
    
    # import pandas as pd
    v_benchmark = pd.read_csv('v_velocity_results.csv',sep = ',')
    u_benchmark = pd.read_csv('u_velocity_results.txt',sep= '\t')
    
    
    horizontal_line = pv_mesh.sample_over_line( Python.tuple(0,L/2,0),Python.tuple(L,L/2,0),resolution= nx)
    v_05 = horizontal_line.point_data['V velocity']
    
    plt.plot(v_benchmark['%x'],v_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(horizontal_line.points[all_slice,0],v_05,label ='lbm Results')
    plt.xlabel('x')
    plt.ylabel('v velocity')
    plt.legend()
    plt.show()
    
    
    vertical_line = pv_mesh.sample_over_line(Python.tuple(L/2,0,0),Python.tuple(L/2,L,0),resolution= ny)
    u_05 = vertical_line.point_data['U velocity']
    
    plt.plot(u_benchmark['%y'],u_benchmark['100'],'o',label = 'Ghia et al')
    plt.plot(vertical_line.points[all_slice,1],u_05,label ='lbm Results')
    plt.xlabel('y')
    plt.ylabel('u velocity')
    plt.legend()
    plt.show()
        

