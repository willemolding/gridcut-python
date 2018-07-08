#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <GridGraph_2D_4C.h>
#include <GridGraph_2D_4C_MT.h>
#include <GridGraph_2D_8C.h>

typedef GridGraph_2D_4C<float, float, float> Grid_2d_4c;
typedef GridGraph_2D_4C_MT<float, float, float> Grid_2d_4c_mt;
typedef GridGraph_2D_8C<float, float, float> Grid_2d_8c;



/*
Extract the solution from a grid as a numpy int array.
Grid must already have been solved for this to work.
Returns a single 1D numpy int array where result[x+y*width] is the binary label assigned to node x,y
*/
template<class grid_type> static PyObject* grid_to_array(grid_type* grid,
                                                         int width, int height)
{
	//create an array to return the result in
	npy_intp outdims[1];
	outdims[0] = width*height;
	PyObject *result = PyArray_SimpleNew(1, outdims, NPY_INT);

	//fill the output array with the results
	for(int i = 0; i < width*height; i++)
	{
		int *data_ptr = (int*)PyArray_GETPTR1(result, i);
		*data_ptr = grid->get_segment(grid->node_id(i % width, i / width));
	}

	return result;
};


/*
Parameters(int width, int height, float pairwise_cost, np float array source, np float array sink)
Source and sink parameters should be a 1D numpy array with width*height elements.
Indexing should be done so that source[x+y*width] is the capacity of node located at (x,y)
*/
static PyObject* gridcut_maxflow_2D_4C_potts(PyObject* self, PyObject *args,
                                             PyObject *keywds)
{
	PyObject *source=NULL, *sink=NULL;
	PyObject *source_arr=NULL, *sink_arr=NULL;
	float pw=0;
	int width, height, n_threads=1, block_size=0;

	//parse arguments
	static char *kwlist[] = {"width", "height", "source", "sink",
	                         "pairwise_cost", "n_threads", "block_size", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "iiOOf|ii", kwlist,
	                                 &width, &height, &source, &sink, &pw,
	                                 &n_threads, &block_size))
        return NULL;

    source_arr = PyArray_FROM_OTF(source, NPY_FLOAT32, NPY_IN_ARRAY);
    sink_arr = PyArray_FROM_OTF(sink, NPY_FLOAT32, NPY_IN_ARRAY);

    if ( source_arr == NULL || sink_arr == NULL )
        return NULL;

    float *srs = (float*)PyArray_DATA(source_arr);
	float *snk = (float*)PyArray_DATA(sink_arr);
	float *edge = new float[width*height];

	//fill edge array with the pairwise weight
	for(int i = 0; i < width*height; i++)
	{
		edge[i] = pw;
	}

	//pass it to the gridcut
	PyObject *result = NULL;

	if(n_threads > 1)
	{
		Grid_2d_4c_mt* grid = new Grid_2d_4c_mt(width, height, n_threads, block_size);
		grid->set_caps(srs, snk, edge, edge, edge, edge);
		grid->compute_maxflow();
		result = grid_to_array<Grid_2d_4c_mt>(grid, width, height);
   		delete grid;
	}
	else
	{
		Grid_2d_4c* grid = new Grid_2d_4c(width, height);
		grid->set_caps(srs, snk, edge, edge, edge, edge);
		grid->compute_maxflow();
		result = grid_to_array<Grid_2d_4c>(grid, width, height);
   		delete grid;
	}

	Py_DECREF(source_arr);
    Py_DECREF(sink_arr);
    delete [] edge;

    return result;
};


static PyObject* gridcut_maxflow_2D_4C(PyObject* self, PyObject *args, PyObject *keywds)
{
	PyObject *source=NULL, *sink=NULL, *up=NULL, *down=NULL, *left=NULL, *right=NULL;
	PyObject *source_arr=NULL, *sink_arr=NULL, *up_arr=NULL, *down_arr=NULL, *left_arr=NULL, *right_arr=NULL;
	int width, height, n_threads=1, block_size=0;

	//parse arguments
	static char *kwlist[] = {"width", "height", "source", "sink", "up", "down",
	                         "left", "right", "n_threads", "block_size", NULL};

    if(!PyArg_ParseTupleAndKeywords(args, keywds, "iiOOOOOO|ii", kwlist,
                                    &width, &height, &source, &sink, &up, &down,
                                    &left, &right, &n_threads, &block_size))
        return NULL;

	source_arr = PyArray_FROM_OTF(source, NPY_FLOAT32, NPY_IN_ARRAY);
    sink_arr = PyArray_FROM_OTF(sink, NPY_FLOAT32, NPY_IN_ARRAY);
    up_arr = PyArray_FROM_OTF(up, NPY_FLOAT32, NPY_IN_ARRAY);
    down_arr = PyArray_FROM_OTF(down, NPY_FLOAT32, NPY_IN_ARRAY);
    left_arr = PyArray_FROM_OTF(left, NPY_FLOAT32, NPY_IN_ARRAY);
    right_arr = PyArray_FROM_OTF(right, NPY_FLOAT32, NPY_IN_ARRAY);

    if (source_arr == NULL || sink_arr == NULL ||
        up_arr==NULL || down_arr==NULL || left_arr == NULL || right_arr == NULL)
        return NULL;

	//print out dim0
	float *srs = (float*)PyArray_DATA(source_arr);
	float *snk = (float*)PyArray_DATA(sink_arr);
	float *u = (float*)PyArray_DATA(up_arr);
	float *d = (float*)PyArray_DATA(down_arr);
	float *l = (float*)PyArray_DATA(left_arr);
	float *r = (float*)PyArray_DATA(right_arr);

	//pass it to the gridcut
	PyObject *result = NULL;

	if(n_threads > 1)
	{
		Grid_2d_4c_mt* grid = new Grid_2d_4c_mt(width, height, n_threads, block_size);
		grid->set_caps(srs, snk, u, d, l, r);
		grid->compute_maxflow();
		result = grid_to_array<Grid_2d_4c_mt>(grid, width, height);
   		delete grid;
	}
	else
	{
		Grid_2d_4c* grid = new Grid_2d_4c(width, height);
		grid->set_caps(srs, snk, u, d, l, r);
		grid->compute_maxflow();
		result = grid_to_array<Grid_2d_4c>(grid, width, height);
   		delete grid;
	}

   	Py_DECREF(source_arr);
    Py_DECREF(sink_arr);
    Py_DECREF(up_arr);
    Py_DECREF(down_arr);
    Py_DECREF(left_arr);
    Py_DECREF(right_arr);

   	return result;
};


static PyObject* gridcut_maxflow_2D_8C_potts(PyObject* self, PyObject *args,
                                             PyObject *keywds)
{
	PyObject *source=NULL, *sink=NULL;
	PyObject *source_arr=NULL, *sink_arr=NULL;
	float pw=0;
	int width, height;


	//parse arguments
	static char *kwlist[] = {"width", "height", "source", "sink",
	                         "pairwise_cost", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "iiOOf", kwlist,
	                                 &width, &height, &source, &sink, &pw))
        return NULL;

    source_arr = PyArray_FROM_OTF(source, NPY_FLOAT32, NPY_IN_ARRAY);
    sink_arr = PyArray_FROM_OTF(sink, NPY_FLOAT32, NPY_IN_ARRAY);

    if ( source_arr == NULL || sink_arr == NULL )
        return NULL;

    float *srs = (float*)PyArray_DATA(source_arr);
	float *snk = (float*)PyArray_DATA(sink_arr);
	float *edge = new float[width*height];
	float *edge_diag = new float[width*height];
    float pw_diag = sqrtf(2 * (pw * pw));

	//fill edge array with the pairwise weight
	for(int i = 0; i < width*height; i++)
	{
		edge[i] = pw;
		edge_diag[i] = pw_diag;
	}

	//pass it to the gridcut
	PyObject *result = NULL;

    Grid_2d_8c* grid = new Grid_2d_8c(width, height);
    grid->set_caps(srs, snk, edge, edge, edge, edge,
                   edge_diag, edge_diag, edge_diag, edge_diag);
    grid->compute_maxflow();
    result = grid_to_array<Grid_2d_8c>(grid, width, height);
    delete grid;

	Py_DECREF(source_arr);
    Py_DECREF(sink_arr);
    delete [] edge;

    return result;
};

// Mapping between python and c function names.
static PyMethodDef gridcutModule_methods[] = {

    {"maxflow_2D_4C", (PyCFunction)gridcut_maxflow_2D_4C,
     METH_VARARGS | METH_KEYWORDS, "maxflow 2D 4C"},

    {"maxflow_2D_4C_potts", (PyCFunction)gridcut_maxflow_2D_4C_potts,
     METH_VARARGS | METH_KEYWORDS, "maxflow 2D 4C potts"},

    {"maxflow_2D_8C_potts", (PyCFunction)gridcut_maxflow_2D_8C_potts,
     METH_VARARGS | METH_KEYWORDS, "maxflow 2D 8C potts"},

    {NULL}
};


#if PY_MAJOR_VERSION >= 3

    // SEE: http://python3porting.com/cextensions.html
    #if PY_MAJOR_VERSION >= 3
        static struct PyModuleDef moduledef = {
            PyModuleDef_HEAD_INIT,
            "gridcut",           /* m_name */
            "Module extension.", /* m_doc */
            -1,                  /* m_size */
            gridcutModule_methods,       /* m_methods */
            NULL,                /* m_reload */
            NULL,                /* m_traverse */
            NULL,                /* m_clear */
            NULL,                /* m_free */
        };
    #endif

    /* Module entrypoint */
    PyMODINIT_FUNC
    PyInit_gridcut(void)
    {
        import_array();

        return PyModule_Create(&moduledef);
    }

#else

    extern "C"
    {
        void initgridcut(void)
        {
            import_array();

            Py_InitModule3("gridcut", gridcutModule_methods,
                           "Module extension.");
        }

    }

#endif
