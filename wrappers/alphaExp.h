#include <Python.h>
#include <numpy/arrayobject.h>
#include <AlphaExpansion_2D_4C.h>
#include <AlphaExpansion_2D_4C_MT.h>
#include <AlphaExpansion_2D_8C.h>

typedef AlphaExpansion_2D_4C<int, float, float> Expansion_2d_4c;
typedef AlphaExpansion_2D_4C_MT<int, float, float> Expansion_2d_4c_mt;
typedef AlphaExpansion_2D_8C<int, float, float> Expansion_2d_8c;


/*
Parameters(int width, int height, float pairwise_cost, np float array source, np float array sink)
Source and sink parameters should be a 1D numpy array with width*height elements.
Indexing should be done so that source[x+y*width] is the capacity of node located at (x,y)
*/
static PyObject* gridcut_expansion_2D_4C_potts(PyObject* self, PyObject *args,
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

//    source_arr = PyArray_FROM_OTF(source, NPY_FLOAT32, NPY_IN_ARRAY);
//    sink_arr = PyArray_FROM_OTF(sink, NPY_FLOAT32, NPY_IN_ARRAY);
//
//    if ( source_arr == NULL || sink_arr == NULL )
//        return NULL;
//
//    float *srs = (float*)PyArray_DATA(source_arr);
//	float *snk = (float*)PyArray_DATA(sink_arr);
//	float *edge = new float[width*height];
//
//	//fill edge array with the pairwise weight
//	for(int i = 0; i < width*height; i++)
//	{
//		edge[i] = pw;
//	}

	//pass it to the gridcut
	PyObject *result = NULL;

//	if(n_threads > 1)
//	{
//		Grid_2d_4c_mt* grid = new Grid_2d_4c_mt(width, height, n_threads, block_size);
//		grid->set_caps(srs, snk, edge, edge, edge, edge);
//		grid->compute_maxflow();
//		result = grid_to_array<Grid_2d_4c_mt>(grid, width, height);
//   		delete grid;
//	}
//	else
//	{
//		Grid_2d_4c* grid = new Grid_2d_4c(width, height);
//		grid->set_caps(srs, snk, edge, edge, edge, edge);
//		grid->compute_maxflow();
//		result = grid_to_array<Grid_2d_4c>(grid, width, height);
//   		delete grid;
//	}
//
//	Py_DECREF(source_arr);
//    Py_DECREF(sink_arr);
//    delete [] edge;

    return result;
};



static PyMethodDef expansion_funcs[] = {

     {"expansion_2D_4C_potts", (PyCFunction)gridcut_expansion_2D_4C_potts,
     METH_VARARGS | METH_KEYWORDS, "a message"},

    {NULL}
};