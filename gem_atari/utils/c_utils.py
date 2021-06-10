import os
import ctypes
import numpy as np

def c_int(value):
    return ctypes.c_int(value)
def c_longlong(value):
    return ctypes.c_longlong(value)
def c_float(value):
    return ctypes.c_float(value)
def c_double(value):
    return ctypes.c_double(value)
def c_ptr(np_array):
    return np.ascontiguousarray(np_array).ctypes.data_as(ctypes.c_char_p)

def c_complie(c_path, so_path=None):
    assert c_path[-2:]=='.c'
    if so_path is None:
        so_path = c_path[:-2]+'.so'
    else:
        assert so_path[-3:]=='.so'
    os.system('gcc -o '+so_path+' -shared -fPIC '+c_path+' -O2')
    return so_path

def cpp_complie(cpp_path, so_path=None):
    # extern "C"
    assert cpp_path[-4:]=='.cpp'
    if so_path is None:
        so_path = cpp_path[:-4]+'.so'
    else:
        assert so_path[-3:]=='.so'
    os.system('gcc -o '+so_path+' -shared -fPIC '+cpp_path+' -O2 -lstdc++')
    return so_path

def load_c_lib(lib_path):
    if lib_path[-2:]=='.c':
        lib_path = c_complie(lib_path)
    elif lib_path[-4:]=='.cpp':
        lib_path = cpp_complie(lib_path)
    else:
        assert lib_path[-3:]=='.so'
    return ctypes.cdll.LoadLibrary(lib_path)