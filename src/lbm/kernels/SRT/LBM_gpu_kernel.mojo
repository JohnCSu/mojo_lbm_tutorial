from std.gpu import block_dim,block_idx,thread_idx,barrier
from layout import TileTensor,LayoutTensor,coord
from layout.tile_tensor import stack_allocation
from layout.tile_layout import Layout,row_major,Coord,TensorLayout
from std.gpu.memory import AddressSpace
from src.lbm.lattice_models import LatticeModel
from src.lbm import LBM_Grid,LBM_Config
from src.lbm.flags import SOLID_NODE,FLUID_NODE
from src.utils import Vector,ContextTileTensor

from std.algorithm.functional import vectorize
from std.sys import simd_width_of

def LBM_kernel[ float_dtype:DType,D:Int,Q:Int,
                lattice_model:LatticeModel[D,Q,float_dtype,DType.int32],
                nx:Int,ny:Int,nz:Int,tile_size:Int,
                //,
                grid: LBM_Grid[lattice_model,nx,ny,nz,tile_size],
                Flayout:Layout[...],
                BClayout:Layout[...],
                Flaglayout:Layout[...],
                config:LBM_Config = LBM_Config(),
                *,
                f_dtype:DType = config.f_dtype.value() if config.f_dtype is not None else float_dtype
                ]
                (
                f_out:TileTensor[f_dtype,type_of(Flayout),MutAnyOrigin],
                f_in:TileTensor[f_dtype,type_of(Flayout),ImmutAnyOrigin],
                bc:TileTensor[float_dtype,type_of(BClayout),ImmutAnyOrigin],
                flags:TileTensor[DType.uint8,type_of(Flaglayout),ImmutAnyOrigin],
                inv_tau:Scalar[float_dtype]
                )
                where tile_size >= 1 and Flayout.rank == 4 and BClayout.rank == 4 and Flaglayout.rank == 3:
    '''
    Base LBM to also handle 3D and non_square Grids. Key assumption is that block dim == tile-size 
    i.e. grid can be non-square but block is squre (same block dim in each x,y,z).
    ''' 
    # Convience Variable Names and constants
    comptime weights = lattice_model.weights
    comptime float_directions = lattice_model.float_directions
    comptime directions = lattice_model.directions
    comptime opposite_index = lattice_model.opposite_indices
    comptime grid_shape:InlineArray[Int,3] = [nx,ny,nz]
    comptime load_f_from_xyzq = load_f[float_dtype,config.use_float16c]

    block_x,block_dim_x = block_idx.x,block_dim.x
    block_y,block_dim_y = block_idx.y,block_dim.y
    block_z,block_dim_z = block_idx.z,block_dim.z

    local_x = thread_idx.x
    local_y = thread_idx.y
    local_z = thread_idx.z
    
    x = block_x*block_dim_x + local_x
    y = block_y*block_dim_y + local_y
    z = block_z*block_dim_z + local_z
    
    index:InlineArray[Int,3] = [x,y,z]    

    # Main Compute
    if (index[0] < grid_shape[0]) and (index[1] < grid_shape[1]) and (index[2] < grid_shape[2]): # Basic Guard
        var f_new = Vector[float_dtype,Q](fill = 0.)
        var velocity = Vector[float_dtype,D](uninitialized = True)
        var rho:Scalar[float_dtype] = 0

        # Streaming Step
        comptime for q in range(Q):
            direction = directions[q]
            pull_index = get_adjacent_idx[D,-1](index,grid_shape,direction) # Pulling Scheme
            pulled_f = load_f_from_xyzq(f_in,pull_index,q)
            f_new[q] = pulled_f

            # BC Step
            pulled_flag = flags.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2])))[0]
            if pulled_flag == SOLID_NODE:
                opp_q = Int(opposite_index[q])
                f_opp = load_f_from_xyzq(f_in,index,opp_q)
                comptime for ii in range(D):
                    velocity[ii] = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],ii)))[0]
                rho = bc.load(coord[DType.uint32]((pull_index[0],pull_index[1],pull_index[2],D)))[0]
                f_new[q] = (f_opp) + 2.*3.*weights[q]*rho*(float_directions[q].dot(velocity))
                
        # Get Velocity and Density
        velocity.fill(0)
        rho = 0
        comptime for q in range(Q):    
            rho += f_new[q]
            velocity += f_new[q]*float_directions[q]

        comptime if config.DDF_shift:
            rho += 1

        velocity /= rho

        # Collision Term
        u_dot_u = velocity.dot(velocity)
        comptime for q in range(Q):
            f_eq = SRT[config.DDF_shift](weights[q],rho,velocity,u_dot_u,float_directions[q])
            f_next = (f_new[q] -  inv_tau*(f_new[q]- f_eq)) # fp
            store_f[config.use_float16c](f_out,f_next,index,q)


@always_inline
def store_f[
        f_dtype:DType,
        FlayoutType:TensorLayout,
        float_dtype:DType,
        //,
        use_float16c:Bool = False,
        ]
        (
        f:TileTensor[f_dtype,FlayoutType,MutAnyOrigin],
        val:Scalar[float_dtype],
        index:InlineArray[Int,3],
        q:Int
        ):
        comptime if use_float16c:
            comptime assert f_dtype == DType.uint16
            f_next = Float32(val)
            f_as_uint = Scalar[f_dtype](LBM_Config.fp32_to_fp16c(f_next))
            f.store(coord = coord[DType.uint32]((index[0],index[1],index[2],q)),value = f_as_uint)
        else:
            f.store(coord = coord[DType.uint32]((index[0],index[1],index[2],q)),value = Scalar[f_dtype](val))


@always_inline
def load_f[
        f_dtype:DType,
        FlayoutType:TensorLayout,
        //,
        float_dtype:DType,
        use_float16c:Bool = False,
        ]
        (
        f:TileTensor[f_dtype,FlayoutType,ImmutAnyOrigin],
        index:InlineArray[Int,3],q:Int
        ) -> Scalar[float_dtype]:
        comptime to_compute_float = Scalar[float_dtype]
        comptime assert FlayoutType.rank == 4, 'For all LBM grids we use i,j,k,q indexing'

        comptime if use_float16c:
                comptime assert f_dtype == DType.uint16, 'Float16C requires the f tiletensors to be uint16 dtype'
                pulled_f = to_compute_float(LBM_Config.fp16c_to_fp32( f.load(coord[DType.uint32]((index[0],index[1],index[2],q)))[0] ))
            else:
                pulled_f =to_compute_float(f.load(coord[DType.uint32]((index[0],index[1],index[2],q)))[0])
        
        return pulled_f


@always_inline
def get_adjacent_idx[D:Int,shift:Int = 1](index:InlineArray[Int,3],grid_shape:InlineArray[Int,3],direction:Vector[DType.int32,D],) -> InlineArray[Int,3]:
    comptime assert D <= 3 
    adj_index = InlineArray[Int,3](fill = 0 )
    comptime for d in range(D):
        adj_index[d] = (index[d] + shift*Int(direction[d])) % grid_shape[d]
    return adj_index

@always_inline
def SRT[dtype:DType,D:Int,//,DDF_shift:Bool = False](weight:Scalar[dtype],density:Scalar[dtype],velocity:Vector[dtype,D],u_dot_u:Scalar[dtype],direction:Vector[dtype,D]) -> Scalar[dtype]:
    comptime assert dtype.is_floating_point(), 'DType to BGK_collision term should be Float point like' # Weied using where statement cause compile error?
    ei_dot_u = velocity.dot(direction)
    comptime if DDF_shift:
        return weight*density*(3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u) +weight*(density - 1)
    else:
        return weight*density*(1 + 3.*ei_dot_u + 4.5*ei_dot_u*ei_dot_u - 1.5*u_dot_u)




@always_inline
def rho_and_u_from_f_vec[float_dtype:DType,Q:Int,D:Int,//,
                        float_directions:InlineArray[Vector[float_dtype,D],Q],
                        weights:Optional[Vector[float_dtype,Q]] = None,
                        DDF_shift:Bool = False]
                        (f_new:Vector[float_dtype,Q],
                        mut velocity:Vector[float_dtype,D],
                        mut rho:Scalar[float_dtype],
                        ):
    comptime assert D <= 3
    velocity.fill(0)
    rho = 0
    comptime if DDF_shift:
        comptime assert DDF_shift and (weights is not None), 'if DDF_shift is specified in config, weights must be passed in'
        # comptime DDF_shift_constant = get_DDF_shift_constant(float_directions,weights.value())  # This might be 0
        comptime for q in range(Q):    
            rho += f_new[q]
            velocity += f_new[q]*float_directions[q]
        rho += 1
        velocity/=rho
    else:
        comptime for q in range(Q):    
            rho += f_new[q]
            velocity += f_new[q]*float_directions[q]
        velocity /= rho