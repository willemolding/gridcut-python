#include <Python.h>
#include <maxFlow.h>
#include <alphaExp.h>

extern "C"
{

	void initgridcut(void)
	{
	    Py_InitModule3("gridcut", maxflow_funcs,
	                   "Extension of maxflow");
	    Py_InitModule3("gridcut", expansion_funcs,
	                   "Extension alpha expansion");
	    import_array();
	}

}