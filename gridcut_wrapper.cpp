#include <Python.h>
#include <numpy/arrayobject.h>
#include <GridGraph_2D_4C.h>

typedef GridGraph_2D_4C<float,float,float> Grid;


///Extract the solution from a grid as a numpy int array.
///Grid must alread have been solved for this to work.
static PyObject* grid_to_array(Grid* grid, int width, int height)
{

	//create an array to return the result in
	npy_intp outdims[2];
	outdims[0] = height;
	outdims[1] = width;
	PyObject *result = PyArray_SimpleNew(2, outdims, NPY_INT);

	//fill the output array with the results
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < height; j++)
		{
			int *data_ptr = (int*)PyArray_GETPTR2(result, j, i);
			*data_ptr = grid->get_segment(grid->node_id(i,j));
		}
	}

	return result;
}


//Parameters(int width, int height, float pairwise_cost, np float array source, np float array sink)
static PyObject* gridcut_solve_2D_4C_potts(PyObject* self, PyObject *args)
{
	PyObject *source=NULL,*sink=NULL;
	PyObject *source_arr=NULL,*sink_arr=NULL;
	int width,height;
	float w;

	if (!PyArg_ParseTuple(args, "iifO!O!",&width,&height,&w, &PyArray_Type, &source,
    	&PyArray_Type, &sink)) return NULL;

    source_arr = PyArray_FROM_OTF(source, NPY_FLOAT32, NPY_IN_ARRAY);
    sink_arr = PyArray_FROM_OTF(sink, NPY_FLOAT32, NPY_IN_ARRAY);

    if ( source_arr == NULL || sink_arr == NULL ) return NULL;

    float *srs = (float*)PyArray_DATA(source_arr);
	float *s = (float*)PyArray_DATA(sink_arr);
	float *edge = new float[width*height];

	//fill edge array with the pairwise weight
	for(int i = 0; i < width*height; i++)
	{
		edge[i] = w;
	}

	//pass it to the gridcut
	Grid* grid = new Grid(width,height);
	grid->set_caps(srs,s,edge,edge,edge,edge);
	grid->compute_maxflow();


	PyObject *result = grid_to_array(grid, width, height);

   	Py_DECREF(source_arr);
    Py_DECREF(sink_arr);
   	delete grid;
    delete [] edge;

    return result;

}


//def gridcut.2D_4C(int width, int height, source, sink, up, down, left, right)
static PyObject* gridcut_solve_2D_4C(PyObject* self, PyObject *args)
{
	PyObject *source=NULL,*sink=NULL,*up=NULL,*down=NULL,*left=NULL,*right=NULL;
	PyObject *source_arr=NULL,*sink_arr=NULL,*up_arr=NULL,*down_arr=NULL,*left_arr=NULL,*right_arr=NULL;
	int width,height;

    if (!PyArg_ParseTuple(args, "iiO!O!O!O!O!O!",&width,&height, &PyArray_Type, &source,
    	&PyArray_Type, &sink,&PyArray_Type, &up,&PyArray_Type, &down,&PyArray_Type, &left,&PyArray_Type, &right)) return NULL;

	source_arr = PyArray_FROM_OTF(source, NPY_FLOAT32, NPY_IN_ARRAY);
    sink_arr = PyArray_FROM_OTF(sink, NPY_FLOAT32, NPY_IN_ARRAY);
    up_arr = PyArray_FROM_OTF(up, NPY_FLOAT32, NPY_IN_ARRAY);
    down_arr = PyArray_FROM_OTF(down, NPY_FLOAT32, NPY_IN_ARRAY);
    left_arr = PyArray_FROM_OTF(left, NPY_FLOAT32, NPY_IN_ARRAY);
    right_arr = PyArray_FROM_OTF(right, NPY_FLOAT32, NPY_IN_ARRAY);

    if ( source_arr == NULL || sink_arr == NULL || up_arr==NULL || down_arr==NULL || left_arr == NULL || right_arr == NULL) return NULL;

	//print out dim0
	float *srs = (float*)PyArray_DATA(source_arr);
	float *s = (float*)PyArray_DATA(sink_arr);
	float *u = (float*)PyArray_DATA(up_arr);
	float *d = (float*)PyArray_DATA(down_arr);
	float *l = (float*)PyArray_DATA(left_arr);
	float *r = (float*)PyArray_DATA(right_arr);


	//pass it to the gridcut
	Grid* grid = new Grid(width,height);
	grid->set_caps(srs,s,u,d,l,r);
	grid->compute_maxflow();


   	Py_DECREF(source_arr);
    Py_DECREF(sink_arr);
    Py_DECREF(up_arr);
    Py_DECREF(down_arr);
    Py_DECREF(left_arr);
    Py_DECREF(right_arr);


    PyObject *result = grid_to_array(grid, width, height);
   	delete grid;
   	return result;
}

static char gridcut_docs[] =
    "helloworld( ): Any message you want to put here!!\n";

static PyMethodDef gridcut_funcs[] = {

    {"solve_2D_4C", (PyCFunction)gridcut_solve_2D_4C, 
     METH_VARARGS, gridcut_docs},

     {"solve_2D_4C_potts", (PyCFunction)gridcut_solve_2D_4C_potts, 
     METH_VARARGS, gridcut_docs},

    {NULL}
};


extern "C"
{

	void initgridcut(void)
	{
	    Py_InitModule3("gridcut", gridcut_funcs,
	                   "Extension module example!");
	    import_array();
	}

}